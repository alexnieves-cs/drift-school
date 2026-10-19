/** Welford running mean / variance for observations and reward scale. */

export class RunningNorm {
  readonly n: number;
  count = 1e-4;
  mean: Float64Array;
  m2: Float64Array;

  constructor(n: number) {
    this.n = n;
    this.mean = new Float64Array(n);
    this.m2 = new Float64Array(n);
    this.m2.fill(1);
  }

  update(x: ArrayLike<number>, stride = this.n): void {
    const rows = Math.floor(x.length / stride);
    for (let r = 0; r < rows; r++) {
      this.count++;
      const base = r * stride;
      for (let i = 0; i < this.n; i++) {
        const v = x[base + i] ?? 0;
        const d = v - this.mean[i]!;
        this.mean[i] = this.mean[i]! + d / this.count;
        this.m2[i] = this.m2[i]! + d * (v - this.mean[i]!);
      }
    }
  }

  std(i: number): number {
    return Math.sqrt(Math.max(this.m2[i]! / this.count, 1e-6));
  }

  apply(x: Float32Array, stride = this.n): void {
    const rows = Math.floor(x.length / stride);
    for (let r = 0; r < rows; r++) {
      const base = r * stride;
      for (let i = 0; i < this.n; i++) {
        x[base + i] = (x[base + i]! - this.mean[i]!) / this.std(i);
      }
    }
  }
}

export class RewardScale {
  count = 1e-4;
  mean = 0;
  m2 = 1;
  update(r: number): void {
    this.count++;
    const d = r - this.mean;
    this.mean += d / this.count;
    this.m2 += d * (r - this.mean);
  }
  scale(r: number): number {
    const s = Math.sqrt(Math.max(this.m2 / this.count, 1e-4));
    return r / s;
  }
}
