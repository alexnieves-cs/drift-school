import { describe, expect, it } from 'vitest';
import { limitCurvature, rdp, resamplePolyline, smoothStroke, toFloat, trimClosingTail, dist, type V2 } from '../../src/editor/stroke';
import { validateTrack } from '../../src/track/build';
import { makeRng } from '../../src/shared/rng';

/** A stadium path sampled every `step` metres, starting at arc length `offset`. */
function stadium(r: number, straight: number, step = 0.12): V2[] {
  const pts: V2[] = [];
  const total = 2 * straight + 2 * Math.PI * r;
  for (let s = 0; s < total; s += step) {
    if (s < straight) pts.push({ x: -straight / 2 + s, z: -r });
    else if (s < straight + Math.PI * r) {
      const a = (s - straight) / r - Math.PI / 2;
      pts.push({ x: straight / 2 + r * Math.cos(a), z: r * Math.sin(a) });
    } else if (s < 2 * straight + Math.PI * r) pts.push({ x: straight / 2 - (s - straight - Math.PI * r), z: r });
    else {
      const a = (s - 2 * straight - Math.PI * r) / r + Math.PI / 2;
      pts.push({ x: -straight / 2 + r * Math.cos(a), z: r * Math.sin(a) });
    }
  }
  return pts;
}

/** A mouse stroke over `loop`: random start, ends 0.5 m short of the start, 0.04 m pixel grid, small hand jitter. */
function stroke(loop: V2[], rng: ReturnType<typeof makeRng>): V2[] {
  const k = rng.int(loop.length);
  const rotated = loop.slice(k).concat(loop.slice(0, k));
  const body = rotated.slice(0, rotated.length - Math.floor(0.5 / 0.12));
  const snap = (v: number) => Math.round((v + (rng.next() - 0.5) * 0.03) / 0.04) * 0.04;
  return body.map((p) => ({ x: snap(p.x), z: snap(p.z) }));
}

describe('stroke processing', () => {
  it('turns closed pixel-quantized stadium strokes into valid tracks', () => {
    const rng = makeRng(5, 'strokes');
    let ok = 0;
    let total = 0;
    for (const [r, straight, hw] of [
      [4, 10, 1],
      [3, 8, 1],
      [5, 14, 1],
      [6, 6, 1.2],
    ] as const) {
      const loop = stadium(r, straight);
      for (let i = 0; i < 30; i++) {
        const raw = stroke(loop, rng);
        expect(dist(raw[0]!, raw[raw.length - 1]!)).toBeLessThan(1);
        const smooth = smoothStroke(raw, true, hw);
        total++;
        if (validateTrack(toFloat([...smooth, { ...smooth[0]! }]), hw).ok) ok++;
      }
    }
    // Before the seam, resample and curvature-limit fixes this was 2 of 40.
    expect(ok / total).toBeGreaterThanOrEqual(0.9);
  });

  it('RDP keeps corners and drops collinear points', () => {
    const line: V2[] = Array.from({ length: 20 }, (_, i) => ({ x: i, z: 0 }));
    expect(rdp(line, 0.01)).toHaveLength(2);
    const corner = [...line, ...Array.from({ length: 10 }, (_, i) => ({ x: 19, z: i + 1 }))];
    expect(rdp(corner, 0.01)).toHaveLength(3);
  });

  it('trims the overlap at the end of a closed stroke', () => {
    const loop = stadium(4, 10);
    const raw = [...loop, ...loop.slice(0, 5)];
    const trimmed = trimClosingTail(raw);
    expect(trimmed.length).toBeLessThan(raw.length);
    expect(dist(trimmed[trimmed.length - 1]!, trimmed[0]!)).toBeGreaterThanOrEqual(1);
  });

  it('resamples to even spacing', () => {
    const out = resamplePolyline(stadium(4, 10, 0.03), 0.3, true);
    const gaps = out.map((p, i) => dist(p, out[(i + 1) % out.length]!));
    expect(Math.max(...gaps)).toBeLessThan(0.31);
    expect(Math.min(...gaps)).toBeGreaterThan(0.2);
  });

  it('curvature limiter flattens a spike and leaves a gentle curve alone', () => {
    const arc = Array.from({ length: 40 }, (_, i) => ({ x: 10 * Math.cos(i / 20), z: 10 * Math.sin(i / 20) }));
    const same = limitCurvature(arc, false, 0.5);
    expect(dist(same[20]!, arc[20]!)).toBeLessThan(1e-9);
    const spiky = arc.map((p, i) => (i === 20 ? { x: p.x + 0.4, z: p.z } : p));
    const fixed = limitCurvature(spiky, false, 0.5);
    expect(dist(fixed[20]!, arc[20]!)).toBeLessThan(dist(spiky[20]!, arc[20]!));
  });
});
