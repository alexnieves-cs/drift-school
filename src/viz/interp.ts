/** 30 Hz pose buffer interpolated for rendering. Each pose is 7 floats: x,y,z,qw,qx,qy,qz. */

export class PoseInterp {
  private prev = new Float32Array(0);
  private next = new Float32Array(0);
  private stamp = 0;
  readonly interval = 1000 / 30;

  push(poses: Float32Array, n: number, now = performance.now()): void {
    const count = n * 7;
    if (this.next.length !== count) {
      this.prev = poses.slice(0, count);
      this.next = poses.slice(0, count);
    } else {
      this.prev.set(this.next);
      this.next.set(poses.subarray(0, count));
    }
    this.stamp = now;
  }

  sample(out: Float32Array, n: number, now = performance.now()): void {
    const t = Math.max(0, Math.min(1, (now - this.stamp) / this.interval));
    const count = Math.min(out.length, n * 7, this.next.length);
    for (let i = 0; i < count; i++) out[i] = this.prev[i]! + (this.next[i]! - this.prev[i]!) * t;
  }
}
