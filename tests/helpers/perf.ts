/**
 * Wall-clock budgets need headroom on shared CI runners and a warmed-up JIT. `PERF_SLACK`
 * multiplies every budget; it defaults to 3 on CI and 1 elsewhere.
 */
export const SLACK = Number(process.env.PERF_SLACK ?? (process.env.CI ? 3 : 1));

/** Run `fn` `warmup` times untimed, then return the fastest of `runs` timed calls in ms. */
export function bestOf(fn: () => void, runs = 7, warmup = 3): number {
  for (let i = 0; i < warmup; i++) fn();
  let best = Infinity;
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    best = Math.min(best, performance.now() - t0);
  }
  return best;
}

export function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}
