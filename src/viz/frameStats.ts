/**
 * Per-frame CPU timing, recorded inside the animation-frame callback. These numbers are the
 * JS side of a frame and do not depend on the GPU: they are what to compare across machines.
 */

export interface FrameSample {
  /** Scene update: interpolation, instance matrices, hero, cameras. */
  updateMs: number;
  /** Time spent inside renderer.render (command submission, not GPU execution). */
  renderMs: number;
  /** DOM, HUD, charts and other panel work. */
  uiMs: number;
  /** Whole animation-frame callback. */
  totalMs: number;
  calls: number;
  triangles: number;
}

export interface Dist {
  mean: number;
  p50: number;
  p95: number;
  max: number;
}

export interface FrameSummary {
  frames: number;
  update: Dist;
  render: Dist;
  ui: Dist;
  total: Dist;
  calls: number;
  triangles: number;
}

const KEYS = ['updateMs', 'renderMs', 'uiMs', 'totalMs'] as const;

export function dist(values: ArrayLike<number>, n = values.length): Dist {
  if (n === 0) return { mean: 0, p50: 0, p95: 0, max: 0 };
  const sorted = Array.from({ length: n }, (_, i) => values[i]!).sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(n - 1, Math.floor(q * n))]!;
  let sum = 0;
  for (const v of sorted) sum += v;
  return { mean: sum / n, p50: at(0.5), p95: at(0.95), max: sorted[n - 1]! };
}

export class FrameStats {
  private readonly cap: number;
  private cols: Record<(typeof KEYS)[number], Float64Array>;
  private count = 0;
  private head = 0;
  private lastCalls = 0;
  private lastTriangles = 0;

  constructor(cap = 900) {
    this.cap = cap;
    this.cols = {
      updateMs: new Float64Array(cap),
      renderMs: new Float64Array(cap),
      uiMs: new Float64Array(cap),
      totalMs: new Float64Array(cap),
    };
  }

  record(s: FrameSample): void {
    for (const k of KEYS) this.cols[k][this.head] = s[k];
    this.head = (this.head + 1) % this.cap;
    this.count = Math.min(this.cap, this.count + 1);
    this.lastCalls = s.calls;
    this.lastTriangles = s.triangles;
  }

  reset(): void {
    this.count = 0;
    this.head = 0;
  }

  summary(): FrameSummary {
    const n = this.count;
    return {
      frames: n,
      update: dist(this.cols.updateMs, n),
      render: dist(this.cols.renderMs, n),
      ui: dist(this.cols.uiMs, n),
      total: dist(this.cols.totalMs, n),
      calls: this.lastCalls,
      triangles: this.lastTriangles,
    };
  }
}
