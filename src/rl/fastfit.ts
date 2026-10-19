import { ACT_DIM, OBS_DIM } from '../shared/types';

/**
 * Plain typed-array backprop + Adam for the policy mean network (48 -> H tanh -> H tanh -> 2),
 * used for the imitation phase only. TF.js's CPU backend needs ~0.35 ms per sample per gradient
 * step for this net; this loop needs ~0.12 ms, which is what lets the whole recipe fit in the
 * 3 minute budget. Weights are copied out of / back into the TF model around each `run`, so the
 * rest of the stack (PPO, act, checkpoints) is unchanged. tests/rl/fastfit.test.ts checks the
 * gradients against TF's autodiff.
 *
 * Loss: mean over batch and action dims of (tanh(mean) - target)^2, same as GaussianPolicy.imitate.
 */
export class FastImitator {
  readonly h: number;
  /** Weights in TF dense layout, [in][out] row-major: W1, b1, W2, b2, W3, b3. */
  readonly params: Float32Array[];
  private readonly m: Float32Array[];
  private readonly v: Float32Array[];
  private readonly g: Float32Array[];
  private t = 0;
  private cap = 0;
  private h1 = new Float32Array(0);
  private h2 = new Float32Array(0);
  private d1 = new Float32Array(0);
  private d2 = new Float32Array(0);
  private dO = new Float32Array(0);

  constructor(h = 128) {
    this.h = h;
    const sizes = [OBS_DIM * h, h, h * h, h, h * ACT_DIM, ACT_DIM];
    this.params = sizes.map((n) => new Float32Array(n));
    this.m = sizes.map((n) => new Float32Array(n));
    this.v = sizes.map((n) => new Float32Array(n));
    this.g = sizes.map((n) => new Float32Array(n));
  }

  load(weights: Float32Array[]): void {
    for (let i = 0; i < 6; i++) this.params[i]!.set(weights[i]!);
  }

  private ensure(bs: number): void {
    if (bs <= this.cap) return;
    this.cap = bs;
    this.h1 = new Float32Array(bs * this.h);
    this.h2 = new Float32Array(bs * this.h);
    this.d1 = new Float32Array(bs * this.h);
    this.d2 = new Float32Array(bs * this.h);
    this.dO = new Float32Array(bs * ACT_DIM);
  }

  /** Dense layer: out = act(inp * w + b), w is [nin][nout]. */
  private static layer(
    inp: Float32Array,
    w: Float32Array,
    b: Float32Array,
    out: Float32Array,
    bs: number,
    nin: number,
    nout: number,
    tanh: boolean,
  ): void {
    for (let r = 0; r < bs; r++) {
      const ro = r * nout;
      const ri = r * nin;
      for (let j = 0; j < nout; j++) out[ro + j] = b[j]!;
      for (let k = 0; k < nin; k++) {
        const v = inp[ri + k]!;
        const wo = k * nout;
        for (let j = 0; j < nout; j++) out[ro + j] = out[ro + j]! + v * w[wo + j]!;
      }
      if (tanh) for (let j = 0; j < nout; j++) out[ro + j] = Math.tanh(out[ro + j]!);
    }
  }

  /** Forward + backward on one batch. Fills `this.g`, returns the MSE loss. */
  gradients(x: Float32Array, y: Float32Array, bs: number): number {
    this.ensure(bs);
    const H = this.h;
    const [W1, b1, W2, b2, W3, b3] = this.params as [Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array];
    const [gW1, gb1, gW2, gb2, gW3, gb3] = this.g as [Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array];
    const { h1, h2, d1, d2, dO } = this;
    FastImitator.layer(x, W1, b1, h1, bs, OBS_DIM, H, true);
    FastImitator.layer(h1, W2, b2, h2, bs, H, H, true);
    // Output layer (linear), then tanh in the loss.
    let loss = 0;
    const scale = 2 / (bs * ACT_DIM);
    for (let r = 0; r < bs; r++) {
      for (let j = 0; j < ACT_DIM; j++) {
        let o = b3[j]!;
        for (let k = 0; k < H; k++) o += h2[r * H + k]! * W3[k * ACT_DIM + j]!;
        const t = Math.tanh(o);
        const e = t - y[r * ACT_DIM + j]!;
        loss += e * e;
        dO[r * ACT_DIM + j] = scale * e * (1 - t * t);
      }
    }
    loss /= bs * ACT_DIM;
    for (const g of this.g) g.fill(0);
    // Layer 3 grads and backprop into h2.
    for (let r = 0; r < bs; r++) {
      for (let j = 0; j < ACT_DIM; j++) gb3[j] = gb3[j]! + dO[r * ACT_DIM + j]!;
      for (let k = 0; k < H; k++) {
        const hv = h2[r * H + k]!;
        let acc = 0;
        for (let j = 0; j < ACT_DIM; j++) {
          const d = dO[r * ACT_DIM + j]!;
          gW3[k * ACT_DIM + j] = gW3[k * ACT_DIM + j]! + hv * d;
          acc += W3[k * ACT_DIM + j]! * d;
        }
        d2[r * H + k] = acc * (1 - hv * hv);
      }
    }
    // Layer 2.
    for (let r = 0; r < bs; r++) {
      const ro = r * H;
      for (let j = 0; j < H; j++) gb2[j] = gb2[j]! + d2[ro + j]!;
      for (let k = 0; k < H; k++) {
        const hv = h1[ro + k]!;
        const wo = k * H;
        let acc = 0;
        for (let j = 0; j < H; j++) {
          const d = d2[ro + j]!;
          gW2[wo + j] = gW2[wo + j]! + hv * d;
          acc += W2[wo + j]! * d;
        }
        d1[ro + k] = acc * (1 - hv * hv);
      }
    }
    // Layer 1.
    for (let r = 0; r < bs; r++) {
      const ro = r * H;
      for (let j = 0; j < H; j++) gb1[j] = gb1[j]! + d1[ro + j]!;
      for (let k = 0; k < OBS_DIM; k++) {
        const xv = x[r * OBS_DIM + k]!;
        const wo = k * H;
        for (let j = 0; j < H; j++) gW1[wo + j] = gW1[wo + j]! + xv * d1[ro + j]!;
      }
    }
    return loss;
  }

  /** Copy of the last gradients (for tests). */
  lastGradients(): Float32Array[] {
    return this.g.map((a) => a.slice());
  }

  /** One Adam step on a batch. */
  step(x: Float32Array, y: Float32Array, bs: number, lr: number): number {
    const loss = this.gradients(x, y, bs);
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - Math.pow(b1, this.t);
    const c2 = 1 - Math.pow(b2, this.t);
    for (let i = 0; i < 6; i++) {
      const p = this.params[i]!;
      const g = this.g[i]!;
      const m = this.m[i]!;
      const v = this.v[i]!;
      for (let k = 0; k < p.length; k++) {
        const gk = g[k]!;
        const mk = b1 * m[k]! + (1 - b1) * gk;
        const vk = b2 * v[k]! + (1 - b2) * gk * gk;
        m[k] = mk;
        v[k] = vk;
        p[k] = p[k]! - (lr * (mk / c1)) / (Math.sqrt(vk / c2) + 1e-8);
      }
    }
    return loss;
  }
}
