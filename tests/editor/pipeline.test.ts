import { describe, expect, it } from 'vitest';
import { closeEnds, processStroke, sCurveStroke, triangleStroke } from '../../src/editor/pipeline';
import { decodeTrackHash, encodeTrackHash, quantizePoints } from '../../src/editor/share';
import { relaxCurvature, resampleUniform, rotateToStraightest, toFloat, type V2 } from '../../src/editor/stroke';
import { buildTrackFromPoints, validateTrack } from '../../src/track/build';
import { simulateBaseline } from '../../src/track/baseline';
import { makeRng } from '../../src/shared/rng';

function jitter(points: V2[], sigma: number, seed: number): V2[] {
  const rng = makeRng(seed, 'jitter');
  return points.map((p) => ({ x: p.x + (rng.next() - 0.5) * 2 * sigma, z: p.z + (rng.next() - 0.5) * 2 * sigma }));
}

function maxCurvature(points: V2[]): number {
  const n = points.length;
  let worst = 0;
  for (let i = 0; i < n; i++) {
    const a = points[(i - 1 + n) % n]!;
    const b = points[i]!;
    const c = points[(i + 1) % n]!;
    const cross = Math.abs((b.x - a.x) * (c.z - b.z) - (b.z - a.z) * (c.x - b.x));
    const d = Math.hypot(b.x - a.x, b.z - a.z) * Math.hypot(c.x - b.x, c.z - b.z) * Math.hypot(c.x - a.x, c.z - a.z);
    if (d > 1e-9) worst = Math.max(worst, (2 * cross) / d);
  }
  return worst;
}

describe('editor stroke pipeline', () => {
  it('turns the M4 S-curve mouse path into a closed, valid track the baseline can lap', () => {
    const result = processStroke(sCurveStroke(), 1, 1.05);
    expect(result.closed).toBe(true);
    expect(result.validation).toEqual({ ok: true, issues: [] });
    expect(result.track).not.toBeNull();
    const track = result.track!;
    expect(track.length).toBeGreaterThan(40);
    expect(track.length).toBeLessThan(70);
    expect(simulateBaseline(track).finished).toBe(true);
  });

  it('smooths the stroke (RDP, Chaikin, curvature limit) instead of passing the raw path through', () => {
    const raw = sCurveStroke();
    const result = processStroke(raw, 1, 1.05);
    // Smoothed controls are uniform 0.4 m resamples, not the 0.2 m mouse samples.
    expect(result.points.length).not.toBe(raw.length);
    expect(maxCurvature(result.points)).toBeLessThan(1 / 1.0);
  });

  it('survives mouse jitter of up to 10 cm', () => {
    for (const [sigma, seed] of [
      [0.02, 1],
      [0.05, 2],
      [0.1, 3],
    ] as const) {
      const result = processStroke(jitter(sCurveStroke(), sigma, seed), 1, 1.05);
      expect(result.validation.ok, `sigma ${sigma}: ${result.validation.issues.map((i) => i.reason).join()}`).toBe(true);
    }
  });

  it('survives a mouse that reports whole pixels (4 cm steps at the editor scale)', () => {
    const snap = (v: number) => Math.round(v / 0.04) * 0.04;
    const result = processStroke(
      sCurveStroke().map((p) => ({ x: snap(p.x), z: snap(p.z) })),
      1,
      1.05,
    );
    expect(result.validation.ok).toBe(true);
    expect(simulateBaseline(result.track!).finished).toBe(true);
  });

  it('accepts a loop whatever point the user started drawing at', () => {
    // The validator wants a straight start line; rotateToStraightest moves it, so the start of
    // the stroke (here the top of a lobe) does not matter.
    const result = processStroke(triangleStroke(), 1, 1.05);
    expect(result.validation.ok).toBe(true);
    expect(simulateBaseline(result.track!).finished).toBe(true);
  });

  it('keeps an open stroke open and says so', () => {
    const open: V2[] = [];
    for (let x = 0; x < 12; x += 0.2) open.push({ x, z: Math.sin(x / 2) });
    const result = processStroke(open, 1, 1.05);
    expect(result.closed).toBe(false);
    expect(result.validation.ok).toBe(false);
    expect(result.validation.issues[0]!.reason).toBe('open');
  });

  it('reports why a bad drawing is bad', () => {
    // A figure of eight crosses itself.
    const eight: V2[] = [];
    for (let k = 0; k < 160; k++) {
      const t = (k / 160) * Math.PI * 2;
      eight.push({ x: 9 * Math.sin(t), z: 5 * Math.sin(2 * t) });
    }
    eight.push({ x: eight[0]!.x + 0.1, z: eight[0]!.z });
    const result = processStroke(eight, 1, 1.05);
    expect(result.validation.ok).toBe(false);
    expect(result.validation.issues.map((i) => i.reason)).toContain('selfIntersect');
    // A tiny loop is too short.
    const tiny: V2[] = [];
    for (let k = 0; k < 40; k++) tiny.push({ x: 1.2 * Math.cos((k / 40) * 6.283), z: 1.2 * Math.sin((k / 40) * 6.283) });
    const small = processStroke(tiny, 1, 1.05);
    expect(small.validation.ok).toBe(false);
  });

  it('closeEnds only appends a seam point when the ends are far apart', () => {
    const near: V2[] = [{ x: 0, z: 0 }, { x: 5, z: 0 }, { x: 5, z: 5 }, { x: 0.5, z: 0.2 }];
    expect(closeEnds(near, true)).toHaveLength(4);
    const far: V2[] = [{ x: 0, z: 0 }, { x: 5, z: 0 }, { x: 5, z: 5 }, { x: 0, z: 3 }];
    expect(closeEnds(far, true)).toHaveLength(5);
    expect(closeEnds(far, false)).toHaveLength(4);
  });
});

describe('stroke helpers', () => {
  it('resampleUniform spaces points evenly around a closed loop', () => {
    const circle: V2[] = [];
    for (let k = 0; k < 90; k++) circle.push({ x: 5 * Math.cos((k / 90) * Math.PI * 2), z: 5 * Math.sin((k / 90) * Math.PI * 2) });
    const out = resampleUniform(circle, 0.5, true);
    const gaps = out.map((p, i) => Math.hypot(p.x - out[(i + 1) % out.length]!.x, p.z - out[(i + 1) % out.length]!.z));
    expect(Math.max(...gaps) - Math.min(...gaps)).toBeLessThan(0.05);
    expect(Math.abs(out.length * 0.5 - 2 * Math.PI * 5)).toBeLessThan(1);
  });

  it('relaxCurvature removes a pinch without moving the rest of the loop much', () => {
    const loop: V2[] = [];
    for (let k = 0; k < 120; k++) {
      const t = (k / 120) * Math.PI * 2;
      loop.push({ x: 8 * Math.cos(t), z: 5 * Math.sin(t) });
    }
    loop[30] = { x: loop[30]!.x + 0.35, z: loop[30]!.z + 0.35 };
    const before = maxCurvature(resampleUniform(loop, 0.4, true));
    const relaxed = relaxCurvature(loop, 1.3);
    expect(maxCurvature(relaxed)).toBeLessThan(before);
    expect(maxCurvature(relaxed)).toBeLessThan(1 / 1.3 + 0.2);
  });

  it('rotateToStraightest starts the loop on a straight', () => {
    const rotated = rotateToStraightest(sCurveStroke());
    const flat = toFloat(rotated);
    expect(validateTrack(flat, 1).ok).toBe(true);
  });
});

describe('URL hash', () => {
  it('round-trips the S-curve to the same track id once points are snapped to mm', () => {
    const points = quantizePoints(toFloat(processStroke(sCurveStroke(), 1, 1.05).points));
    const before = buildTrackFromPoints(points, { halfWidth: 1, mu: 1.05, source: 'user' });
    const decoded = decodeTrackHash(encodeTrackHash(points, 1, 1.05));
    expect(decoded.points.length).toBe(points.length);
    for (let i = 0; i < points.length; i++) expect(Math.abs(decoded.points[i]! - points[i]!)).toBeLessThanOrEqual(0.001);
    const after = buildTrackFromPoints(decoded.points, { halfWidth: decoded.halfWidth, mu: decoded.mu, source: 'user' });
    expect(after.id).toBe(before.id);
    expect(after.length).toBeCloseTo(before.length, 6);
  });
});
