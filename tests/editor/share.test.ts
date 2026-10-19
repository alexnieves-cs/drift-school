import { describe, expect, it } from 'vitest';
import { deflateSync } from 'fflate';
import { decodeTrackHash, encodeTrackHash } from '../../src/editor/share';

const b64url = (bytes: Uint8Array) => Buffer.from(bytes).toString('base64url');

describe('track share hash', () => {
  it('round-trips control points within 1 mm', () => {
    const points = new Float32Array([0, 0, 12.3456, -3.2, 8.1, 9.876, -4.25, 6, 0.4, 0.1]);
    const hash = encodeTrackHash(points, 1.25, 0.9);
    const back = decodeTrackHash(hash);
    expect(back.halfWidth).toBeCloseTo(1.25, 5);
    expect(back.mu).toBeCloseTo(0.9, 5);
    expect(back.points.length).toBe(points.length);
    for (let i = 0; i < points.length; i++) expect(Math.abs(back.points[i]! - points[i]!)).toBeLessThanOrEqual(0.001);
  });

  it('rejects hostile or malformed hashes quickly instead of hanging or allocating', () => {
    const square = Float32Array.from([-900, -900, 900, -900, 900, 900, -900, 900]);
    const payloads: Record<string, string> = {
      empty: '',
      garbage: '!!!!',
      'deflate bomb inside the length cap': b64url(deflateSync(new Uint8Array(2_000_000))),
      'deflate bomb over the length cap': b64url(deflateSync(new Uint8Array(64_000_000))),
      'coordinates far outside the world': encodeTrackHash(square, 1, 1),
      'NaN half-width': encodeTrackHash(Float32Array.from([0, 0, 8, 0, 8, 8, 0, 8]), NaN, 1),
      'huge half-width': encodeTrackHash(Float32Array.from([0, 0, 8, 0, 8, 8, 0, 8]), 1e4, 1),
      'too few points': encodeTrackHash(Float32Array.from([0, 0, 1, 1]), 1, 1),
    };
    for (const [name, hash] of Object.entries(payloads)) {
      const t0 = performance.now();
      expect(() => decodeTrackHash(hash), name).toThrow();
      expect(performance.now() - t0, name).toBeLessThan(500);
    }
  });
});
