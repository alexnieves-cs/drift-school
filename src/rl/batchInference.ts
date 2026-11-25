/**
 * Batched policy inference: gather the observation rows that several callers need in one tick
 * (training slots, ghosts, the hero car, eval cars), run them through the network as one forward
 * pass, and hand each caller back its own rows in order.
 *
 * Rows are padded with zeros up to a fixed bucket size. On a GPU backend every new input shape
 * costs a shader compile, so a few fixed shapes keep that cost bounded; on CPU the padding only
 * costs a few wasted rows. Batches larger than `maxBatch` are split into chunks. The dense layers
 * treat rows independently, so padding and chunking do not change any caller's result (tested on
 * the CPU backend in tests/rl/batchInference.test.ts).
 */
import * as tf from '@tensorflow/tfjs';
import { ACT_DIM, OBS_DIM } from '../shared/types';
import type { GaussianPolicy } from './model';

/** One forward pass over `rows` rows of `inDim` floats, returning `rows * outDim` floats. */
export type ForwardFn = (x: Float32Array, rows: number) => Float32Array;

export interface BatchOpts {
  /** Allowed padded batch sizes, ascending. Default 1, 8, 32, 128, 512. */
  buckets?: number[];
  /** Largest rows per forward pass. Default: the largest bucket. */
  maxBatch?: number;
}

export interface BatchStats {
  /** Forward passes run. */
  calls: number;
  /** Real rows served. */
  rows: number;
  /** Zero rows added to reach a bucket size. */
  padded: number;
  /** Distinct padded shapes used, which bounds GPU shader compiles. */
  shapes: number[];
}

interface Pending {
  obs: Float32Array;
  rows: number;
  resolve: (out: Float32Array) => void;
  reject: (err: unknown) => void;
}

export const DEFAULT_BUCKETS = [1, 8, 32, 128, 512];

export class BatchedInference {
  readonly inDim: number;
  readonly outDim: number;
  readonly buckets: number[];
  readonly maxBatch: number;
  readonly stats: BatchStats = { calls: 0, rows: 0, padded: 0, shapes: [] };
  private readonly forward: ForwardFn;
  private queue: Pending[] = [];

  constructor(forward: ForwardFn, inDim: number, outDim: number, opts: BatchOpts = {}) {
    const buckets = (opts.buckets ?? DEFAULT_BUCKETS).slice().sort((a, b) => a - b);
    if (buckets.length === 0 || buckets.some((b) => !Number.isInteger(b) || b < 1)) throw new Error('batch buckets must be positive integers');
    this.maxBatch = opts.maxBatch ?? buckets[buckets.length - 1]!;
    if (!Number.isInteger(this.maxBatch) || this.maxBatch < 1) throw new Error('maxBatch must be a positive integer');
    this.buckets = buckets.filter((b) => b <= this.maxBatch);
    if (this.buckets.length === 0) this.buckets = [this.maxBatch];
    this.forward = forward;
    this.inDim = inDim;
    this.outDim = outDim;
  }

  /** Run every request now, as few forward passes as the bucket and chunk sizes allow. */
  run(requests: { obs: Float32Array; rows: number }[]): Float32Array[] {
    let total = 0;
    for (const r of requests) {
      if (!Number.isInteger(r.rows) || r.rows < 0) throw new Error(`batch: rows must be a non-negative integer, got ${r.rows}`);
      if (r.obs.length < r.rows * this.inDim) throw new Error(`batch: expected ${r.rows * this.inDim} inputs, got ${r.obs.length}`);
      total += r.rows;
    }
    const outs = requests.map((r) => new Float32Array(r.rows * this.outDim));
    if (total === 0) return outs;
    // Walk the requests in order, filling chunks of at most maxBatch rows.
    let req = 0;
    let row = 0;
    while (req < requests.length) {
      const take: { req: number; from: number; count: number }[] = [];
      let size = 0;
      while (req < requests.length && size < this.maxBatch) {
        const r = requests[req]!;
        const count = Math.min(r.rows - row, this.maxBatch - size);
        if (count > 0) take.push({ req, from: row, count });
        size += count;
        row += count;
        if (row >= r.rows) {
          req++;
          row = 0;
        }
      }
      if (size === 0) break;
      // Pad up to the smallest bucket that fits. When maxBatch is above every bucket, run unpadded.
      const shape = this.buckets.find((b) => b >= size) ?? size;
      const x = new Float32Array(shape * this.inDim);
      let at = 0;
      for (const t of take) {
        const r = requests[t.req]!;
        x.set(r.obs.subarray(t.from * this.inDim, (t.from + t.count) * this.inDim), at * this.inDim);
        at += t.count;
      }
      const y = this.forward(x, shape);
      if (y.length < shape * this.outDim) throw new Error(`batch: forward returned ${y.length} outputs, expected ${shape * this.outDim}`);
      at = 0;
      for (const t of take) {
        outs[t.req]!.set(y.subarray(at * this.outDim, (at + t.count) * this.outDim), t.from * this.outDim);
        at += t.count;
      }
      this.stats.calls++;
      this.stats.rows += size;
      this.stats.padded += shape - size;
      if (!this.stats.shapes.includes(shape)) this.stats.shapes.push(shape);
    }
    return outs;
  }

  /** Queue rows for the next flush. The promise resolves with this request's outputs. */
  request(obs: Float32Array, rows: number): Promise<Float32Array> {
    return new Promise((resolve, reject) => this.queue.push({ obs, rows, resolve, reject }));
  }

  get pending(): number {
    return this.queue.length;
  }

  /** Serve everything queued so far with one batched run. */
  flush(): void {
    const batch = this.queue;
    this.queue = [];
    if (batch.length === 0) return;
    let outs: Float32Array[];
    try {
      outs = this.run(batch);
    } catch (err) {
      for (const p of batch) p.reject(err);
      return;
    }
    batch.forEach((p, i) => p.resolve(outs[i]!));
  }
}

/** Deterministic mean action, tanh(pi(x)), the same numbers as GaussianPolicy.actMean. */
export function meanActionForward(policy: GaussianPolicy): ForwardFn {
  return (x, rows) => {
    const out = new Float32Array(rows * ACT_DIM);
    tf.tidy(() => {
      out.set(tf.tanh(policy.pi.apply(tf.tensor2d(x, [rows, OBS_DIM])) as tf.Tensor).dataSync() as Float32Array);
    });
    return out;
  };
}

/** Batched mean actions for a policy: OBS_DIM floats in, ACT_DIM floats out per row. */
export function batchedMeanActions(policy: GaussianPolicy, opts: BatchOpts = {}): BatchedInference {
  return new BatchedInference(meanActionForward(policy), OBS_DIM, ACT_DIM, opts);
}
