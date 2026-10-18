import { describe, expect, it } from 'vitest';
import { ReplayPlayer, TAPE_HZ, poseAt, tapeFrames } from '../../src/viz/replay';

function tape(n: number): Float32Array {
  const out = new Float32Array(n * 7);
  for (let i = 0; i < n; i++) {
    const yaw = i * 0.1;
    out.set([i * 0.2, 0.05, -i * 0.1, Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0], i * 7);
  }
  return out;
}

describe('replay', () => {
  it('samples the tape linearly and clamps at both ends', () => {
    const t = tape(10);
    expect(tapeFrames(t)).toBe(10);
    const a = poseAt(t, 0);
    expect(a.x).toBe(0);
    const mid = poseAt(t, 0.5 / TAPE_HZ);
    expect(mid.x).toBeCloseTo(0.1, 6);
    expect(mid.z).toBeCloseTo(-0.05, 6);
    expect(mid.yaw).toBeCloseTo(0.05, 6);
    const end = poseAt(t, 100);
    expect(end.x).toBeCloseTo(9 * 0.2, 6);
  });

  it('turns the short way round across the yaw wrap', () => {
    const t = new Float32Array(14);
    const y0 = Math.PI - 0.05;
    const y1 = -Math.PI + 0.05;
    t.set([0, 0, 0, Math.cos(y0 / 2), 0, Math.sin(y0 / 2), 0], 0);
    t.set([0, 0, 0, Math.cos(y1 / 2), 0, Math.sin(y1 / 2), 0], 7);
    const mid = poseAt(t, 0.5 / TAPE_HZ);
    expect(Math.abs(Math.abs(mid.yaw) - Math.PI)).toBeLessThan(1e-4);
  });

  it('plays the same path every time and then stops', () => {
    const t = tape(20);
    const run = () => {
      const p = new ReplayPlayer();
      expect(p.start(3, t)).toBe(true);
      const out: number[] = [];
      for (let pose = p.advance(1 / 30); pose; pose = p.advance(1 / 30)) out.push(pose.x, pose.y, pose.z, pose.yaw);
      expect(p.active).toBe(false);
      return out;
    };
    const a = run();
    const b = run();
    expect(a.length).toBeGreaterThan(20 * 4);
    expect(b).toEqual(a);
  });

  it('refuses a tape too short to play', () => {
    const p = new ReplayPlayer();
    expect(p.start(1, tape(1))).toBe(false);
    expect(p.active).toBe(false);
    expect(p.advance(0.1)).toBeNull();
  });
});
