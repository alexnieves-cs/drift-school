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

export function decodeTrackHash(hash: string): { points: Float32Array; halfWidth: number; mu: number } {
  const raw = inflateSync(unb64url(hash));
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const halfWidth = view.getFloat32(0, true);
  const mu = view.getFloat32(4, true);
  const n = (raw.byteLength - 8) / 4;
  const points = new Float32Array(n);
  for (let i = 0; i < n; i++) points[i] = view.getInt32(8 + i * 4, true) / 1000;
  return { points, halfWidth, mu };
}
