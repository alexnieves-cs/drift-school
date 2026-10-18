/**
 * Pose stream interpolation. The producer sends poses at its own cadence (30 Hz target, but
 * a PPO update can be seconds apart), the renderer runs at display rate and blends the last
 * two snapshots. Each pose is 7 floats: x, y, z, qw, qx, qy, qz.
 *
 * Nothing here allocates after the first push of a given car count: both snapshots live in
 * persistent buffers and `writeMatrices` fills a caller-owned Float32Array in place.
 */

export const POSE_STRIDE = 7;

/** A car that moves further than this between two snapshots was reset; do not blend across it. */
const TELEPORT_M = 0.75;
const MIN_INTERVAL_MS = 1000 / 60;
const MAX_INTERVAL_MS = 250;

export class PoseInterp {
  private prev = new Float32Array(0);
  private next = new Float32Array(0);
  private snapped = new Uint8Array(0);
  private stamp = 0;
  private count = 0;
  private pushes = 0;
  /** Blend span in ms: tracks the observed push cadence, clamped so a slow producer still glides. */
  interval = 1000 / 30;

  get n(): number {
    return this.count;
  }

  push(poses: Float32Array, n: number, now = performance.now()): void {
    const len = Math.min(n * POSE_STRIDE, poses.length);
    const count = Math.floor(len / POSE_STRIDE);
    if (count !== this.count) {
      this.prev = new Float32Array(count * POSE_STRIDE);
      this.next = new Float32Array(count * POSE_STRIDE);
      this.snapped = new Uint8Array(count);
      this.next.set(poses.subarray(0, count * POSE_STRIDE));
      this.prev.set(this.next);
      this.count = count;
      this.pushes = 0;
    } else {
      this.prev.set(this.next);
      this.next.set(poses.subarray(0, len));
      for (let i = 0; i < count; i++) {
        const o = i * POSE_STRIDE;
        const dx = this.next[o]! - this.prev[o]!;
        const dz = this.next[o + 2]! - this.prev[o + 2]!;
        const jump = dx * dx + dz * dz > TELEPORT_M * TELEPORT_M;
        this.snapped[i] = jump ? 1 : 0;
      }
      const gap = now - this.stamp;
      if (this.pushes > 1 && gap > 0) {
        const clamped = Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, gap));
        this.interval += (clamped - this.interval) * 0.5;
      }
    }
    this.pushes++;
    this.stamp = now;
  }

  /** Blend factor for `now`, 0 at the last push and 1 one interval later. */
  alpha(now = performance.now()): number {
    const t = (now - this.stamp) / this.interval;
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }

  /** Copy the interpolated pose of car `i` into out[0..6]. */
  samplePose(i: number, out: Float32Array, now = performance.now()): void {
    const t = this.alpha(now);
    const o = i * POSE_STRIDE;
    const k = this.snapped[i] ? 1 : t;
    for (let j = 0; j < POSE_STRIDE; j++) out[j] = this.prev[o + j]! + (this.next[o + j]! - this.prev[o + j]!) * k;
  }

  /**
   * Write one instance matrix (16 floats, column major, yaw-only rotation) per car into `out`.
   * The renderer's yaw convention is the negative of the sim's, so qy is negated here.
   * Car `skip` gets a zero-scale matrix so it stays out of the instanced draw but keeps its slot.
   */
  writeMatrices(out: Float32Array, now = performance.now(), skip = -1, minY = 0.04): number {
    const t = this.alpha(now);
    const n = Math.min(this.count, Math.floor(out.length / 16));
    const prev = this.prev;
    const next = this.next;
    for (let i = 0; i < n; i++) {
      const o = i * POSE_STRIDE;
      const m = i * 16;
      if (i === skip) {
        out.fill(0, m, m + 16);
        out[m + 15] = 1;
        continue;
      }
      const k = this.snapped[i] ? 1 : t;
      const x = prev[o]! + (next[o]! - prev[o]!) * k;
      const y = prev[o + 1]! + (next[o + 1]! - prev[o + 1]!) * k;
      const z = prev[o + 2]! + (next[o + 2]! - prev[o + 2]!) * k;
      const qw = prev[o + 3]! + (next[o + 3]! - prev[o + 3]!) * k;
      const qy = -(prev[o + 5]! + (next[o + 5]! - prev[o + 5]!) * k);
      const n2 = qw * qw + qy * qy || 1;
      const c = (qw * qw - qy * qy) / n2;
      const s = (2 * qw * qy) / n2;
      out[m] = c;
      out[m + 1] = 0;
      out[m + 2] = -s;
      out[m + 3] = 0;
      out[m + 4] = 0;
      out[m + 5] = 1;
      out[m + 6] = 0;
      out[m + 7] = 0;
      out[m + 8] = s;
      out[m + 9] = 0;
      out[m + 10] = c;
      out[m + 11] = 0;
      out[m + 12] = x;
      out[m + 13] = y < minY ? minY : y;
      out[m + 14] = z;
      out[m + 15] = 1;
    }
    return n;
  }
}
