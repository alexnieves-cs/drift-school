import { deflateSync, inflateSync } from 'fflate';

function b64url(bytes: Uint8Array): string {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]!);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function unb64url(s: string): Uint8Array {
  const pad = s.length % 4 === 0 ? '' : '='.repeat(4 - (s.length % 4));
  const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/') + pad);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Snap control points to whole millimeters. The learner builds the user track from the
 * snapped points, so the track id (a hash of its samples) is the same before and after a
 * round trip through the URL hash.
 */
export function quantizePoints(points: ArrayLike<number>): Float32Array {
  const out = new Float32Array(points.length);
  for (let i = 0; i < points.length; i++) out[i] = Math.round(points[i]! * 1000) / 1000;
  return out;
}

/** Quantize to millimeters so a round trip stays inside 1 mm. */
export function encodeTrackHash(points: Float32Array, halfWidth: number, mu: number): string {
  const n = points.length;
  const payload = new Uint8Array(8 + n * 4);
  const view = new DataView(payload.buffer);
  view.setFloat32(0, halfWidth, true);
  view.setFloat32(4, mu, true);
  for (let i = 0; i < n; i++) view.setInt32(8 + i * 4, Math.round(points[i]! * 1000), true);
  return b64url(deflateSync(payload));
}

/** Largest accepted hash, decoded payload and control-point count. A drawn track is a few hundred points at most. */
const MAX_HASH_CHARS = 8192;
const MAX_PAYLOAD_BYTES = 8 + 4 * 2 * 1000;
const MAX_COORD_M = 200;

export function decodeTrackHash(hash: string): { points: Float32Array; halfWidth: number; mu: number } {
  if (hash.length > MAX_HASH_CHARS) throw new Error('track hash too long');
  // `size` caps the output buffer, so a deflate bomb cannot expand past it.
  const raw = unzlibCapped(unb64url(hash), MAX_PAYLOAD_BYTES);
  if (raw.byteLength < 8 + 3 * 8 || (raw.byteLength - 8) % 8 !== 0) throw new Error('track hash has the wrong size');
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const halfWidth = view.getFloat32(0, true);
  const mu = view.getFloat32(4, true);
  if (!(halfWidth >= 0.1 && halfWidth <= 5) || !(mu >= 0.1 && mu <= 3)) throw new Error('track hash has bad width or grip');
  const n = (raw.byteLength - 8) / 4;
  const points = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = view.getInt32(8 + i * 4, true) / 1000;
    if (Math.abs(v) > MAX_COORD_M) throw new Error('track hash has a coordinate out of range');
    points[i] = v;
  }
  return { points, halfWidth, mu };
}

function unzlibCapped(bytes: Uint8Array, cap: number): Uint8Array {
  const out = new Uint8Array(cap + 1);
  const raw = inflateSync(bytes, { out });
  if (raw.byteLength > cap) throw new Error('track hash expands too far');
  return raw;
}
