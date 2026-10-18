import { describe, expect, it } from 'vitest';
import { FrameStats, dist } from '../../src/viz/frameStats';
import { QUALITY, pixelRatioFor } from '../../src/viz/renderer';

describe('frame stats', () => {
  it('summarises a ring of samples', () => {
    const s = new FrameStats(10);
    for (let i = 1; i <= 25; i++) s.record({ updateMs: i, renderMs: 1, uiMs: 0.5, totalMs: i + 1.5, calls: 8, triangles: 2000 });
    const sum = s.summary();
    expect(sum.frames).toBe(10);
    // Only the last 10 samples (16..25) survive.
    expect(sum.update.max).toBe(25);
    expect(sum.update.mean).toBeCloseTo(20.5, 6);
    expect(sum.calls).toBe(8);
    s.reset();
    expect(s.summary().frames).toBe(0);
  });

  it('dist handles empty and single-valued input', () => {
    expect(dist([], 0).max).toBe(0);
    expect(dist([4], 1)).toEqual({ mean: 4, p50: 4, p95: 4, max: 4 });
  });
});

describe('pixel budget', () => {
  it('leaves a 720p view at the native ratio on Low and High', () => {
    expect(pixelRatioFor(1280, 720, 1, QUALITY.low)).toBe(1);
    expect(pixelRatioFor(1280, 720, 1, QUALITY.high)).toBe(1);
  });

  it('caps a retina 4K-class view by pixel count, not by shrinking the view', () => {
    const r = pixelRatioFor(1920, 1080, 2, QUALITY.low);
    expect(r).toBeLessThan(1);
    expect(1920 * 1080 * r * r).toBeLessThanOrEqual(QUALITY.low.maxPixels * 1.001);
    const h = pixelRatioFor(2560, 1440, 2, QUALITY.high);
    expect(h).toBeLessThanOrEqual(1.5);
    expect(2560 * 1440 * h * h).toBeLessThanOrEqual(QUALITY.high.maxPixels * 1.001);
  });

  it('never goes below a floor', () => {
    expect(pixelRatioFor(8000, 8000, 3, QUALITY.low)).toBeGreaterThanOrEqual(0.5);
  });
});
