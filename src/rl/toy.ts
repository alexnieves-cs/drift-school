import { OBS_DIM } from '../shared/types';

/** Point-mass reach task. Solved when the particle sits near the origin. */
export class ToyEnv {
  x = 0.9;
  v = 0;
  t = 0;

  reset(): Float32Array {
    this.x = 0.9;
    this.v = 0;
    this.t = 0;
    return this.obs();
  }

  obs(): Float32Array {
    const o = new Float32Array(OBS_DIM);
    o[0] = this.x;
    o[1] = this.v;
    o[2] = 1;
    return o;
  }

  step(action: number): { obs: Float32Array; reward: number; done: boolean } {
    const a = Math.max(-1, Math.min(1, action));
    this.v = this.v * 0.92 + a * 0.18;
    this.x += this.v * 0.08;
    this.t++;
    const reward = -Math.abs(this.x) - 0.01 * a * a;
    const done = this.t >= 40 || Math.abs(this.x) < 0.05;
    return { obs: this.obs(), reward, done };
  }
}
