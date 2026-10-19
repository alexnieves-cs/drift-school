/**
 * Seeded sfc32 PRNG. No Math.random anywhere in sim/track/rl.
 * fork(label) derives an independent stream from the parent seed + label.
 */

export interface RNG {
  /** Uniform float in [0, 1). */
  next(): number;
  nextU32(): number;
  /** Signed unit float in [-1, 1). */
  signed(): number;
  range(min: number, max: number): number;
  int(maxExclusive: number): number;
  fork(label: string): RNG;
  getState(): [number, number, number, number];
}

function xmur3(str: string): () => number {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return h >>> 0;
  };
}

function makeSfc32(a0: number, b0: number, c0: number, d0: number): {
  nextU32: () => number;
  getState: () => [number, number, number, number];
} {
  let a = a0 | 0;
  let b = b0 | 0;
  let c = c0 | 0;
  let d = d0 | 0;
  const nextU32 = (): number => {
    a |= 0;
    b |= 0;
    c |= 0;
    d |= 0;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = ((c << 21) | (c >>> 11)) | 0;
    c = (c + t) | 0;
    return t >>> 0;
  };
  return {
    nextU32,
    getState: () => [a >>> 0, b >>> 0, c >>> 0, d >>> 0],
  };
}

export function makeRng(seed: number, stream = 'root'): RNG {
  const hash = xmur3(`${seed >>> 0}:${stream}`);
  const sfc = makeSfc32(hash(), hash(), hash(), hash());
  const rng: RNG = {
    nextU32: sfc.nextU32,
    next: () => sfc.nextU32() / 4294967296,
    signed: () => sfc.nextU32() / 2147483648 - 1,
    range: (min, max) => min + (max - min) * (sfc.nextU32() / 4294967296),
    int: (maxExclusive) => (maxExclusive <= 0 ? 0 : sfc.nextU32() % maxExclusive),
    fork: (label: string) => makeRng(seed, `${stream}/${label}`),
    getState: sfc.getState,
  };
  return rng;
}

/** Gaussian via Box-Muller using the seeded stream. */
export function randn(rng: RNG): number {
  const u = Math.max(rng.next(), 1e-12);
  const v = rng.next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}
