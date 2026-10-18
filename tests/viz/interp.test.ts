import { describe, expect, it } from 'vitest';
import { PoseInterp, POSE_STRIDE } from '../../src/viz/interp';

function pose(x: number, z: number, yaw: number): number[] {
  return [x, 0.05, z, Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0];
}

function pack(cars: number[][]): Float32Array {
  return Float32Array.from(cars.flat());
}

describe('PoseInterp', () => {
  it('blends two snapshots over one interval and clamps after it', () => {
    const it0 = new PoseInterp();
    it0.push(pack([pose(0, 0, 0)]), 1, 1000);
    it0.push(pack([pose(0.3, 0.4, 0)]), 1, 1000 + it0.interval);
    const out = new Float32Array(16);
    it0.writeMatrices(out, 1000 + it0.interval, -1);
    expect(out[12]).toBeCloseTo(0, 5);
    expect(out[14]).toBeCloseTo(0, 5);
    it0.writeMatrices(out, 1000 + it0.interval * 1.5, -1);
    expect(out[12]).toBeCloseTo(0.15, 5);
    expect(out[14]).toBeCloseTo(0.2, 5);
    it0.writeMatrices(out, 1000 + it0.interval * 5, -1);
    expect(out[12]).toBeCloseTo(0.3, 5);
    expect(out[14]).toBeCloseTo(0.4, 5);
  });

  it('writes a yaw-only rotation that matches the renderer convention', () => {
    const it0 = new PoseInterp();
    const yaw = 0.7;
    const snap = pack([pose(3, 4, yaw)]);
    it0.push(snap, 1, 0);
    const out = new Float32Array(16);
    it0.writeMatrices(out, 0, -1);
    // Rotation about +Y by -yaw: [c 0 s; 0 1 0; -s 0 c] in row terms, column-major in `out`.
    const c = Math.cos(-yaw);
    const s = Math.sin(-yaw);
    expect(out[0]).toBeCloseTo(c, 5);
    expect(out[2]).toBeCloseTo(-s, 5);
    expect(out[8]).toBeCloseTo(s, 5);
    expect(out[10]).toBeCloseTo(c, 5);
    expect(out[5]).toBe(1);
    expect(out[15]).toBe(1);
  });

  it('keeps a skipped car out of the draw without shifting the others', () => {
    const it0 = new PoseInterp();
    it0.push(pack([pose(0, 0, 0), pose(5, 5, 0), pose(9, 9, 0)]), 3, 0);
    const out = new Float32Array(16 * 3);
    const n = it0.writeMatrices(out, 0, 0);
    expect(n).toBe(3);
    expect(out[0]).toBe(0);
    expect(out[5]).toBe(0);
    expect(out[12 + 16]).toBeCloseTo(5, 5);
    expect(out[12 + 32]).toBeCloseTo(9, 5);
  });

  it('does not blend across a reset (teleport)', () => {
    const it0 = new PoseInterp();
    it0.push(pack([pose(0, 0, 0)]), 1, 0);
    it0.push(pack([pose(20, 0, 0)]), 1, 33);
    const out = new Float32Array(16);
    it0.writeMatrices(out, 33, -1);
    expect(out[12]).toBeCloseTo(20, 5);
  });

  it('adapts the blend span to a slow producer but clamps it', () => {
    const it0 = new PoseInterp();
    let t = 0;
    for (let i = 0; i < 8; i++) {
      it0.push(pack([pose(i * 0.1, 0, 0)]), 1, t);
      t += 5000;
    }
    expect(it0.interval).toBeLessThanOrEqual(250);
    expect(it0.interval).toBeGreaterThan(200);
  });

  it('resizes when the car count changes', () => {
    const it0 = new PoseInterp();
    it0.push(pack([pose(1, 1, 0)]), 1, 0);
    it0.push(pack([pose(1, 1, 0), pose(2, 2, 0)]), 2, 33);
    expect(it0.n).toBe(2);
    const out = new Float32Array(32);
    expect(it0.writeMatrices(out, 33, -1)).toBe(2);
    expect(out[12 + 16]).toBeCloseTo(2, 5);
    expect(POSE_STRIDE).toBe(7);
  });
});
