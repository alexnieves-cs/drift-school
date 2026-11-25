/**
 * TF.js backend choice for inference: try the fast backends in order and fall back to CPU.
 *
 * Training stays on CPU (`ensureBackend` in model.ts) so runs are reproducible; this module is for
 * the inference path, where a GPU backend can run one large batched forward pass per tick. A
 * backend only counts as working once a small matmul on it matches the same matmul done in plain
 * JavaScript, because a backend can initialise and still compute wrong numbers (for example, half
 * precision on some mobile GPUs).
 *
 * Exercised so far, in Node only: the CPU path; WebGL and WebGPU requests falling back to CPU
 * because neither is registered in Node; and stand-in factories that fail to initialise. The WebGPU
 * package (@tensorflow/tfjs-backend-webgpu) is not a dependency, so in a browser a WebGPU request
 * reports "not registered" and WebGL is tried next. Neither GPU backend has run on a GPU.
 */
import * as tf from '@tensorflow/tfjs';

export type BackendName = 'webgpu' | 'webgl' | 'cpu';

export interface BackendAttempt {
  name: BackendName;
  ok: boolean;
  /** Why the backend was skipped, when it was. */
  reason?: string;
}

export interface BackendChoice {
  /** The backend TF.js is now using. */
  active: string;
  attempts: BackendAttempt[];
  /** True when the first choice was not the one that worked. */
  fellBack: boolean;
}

export const DEFAULT_ORDER: BackendName[] = ['webgpu', 'webgl', 'cpu'];

/** Largest absolute error the smoke test accepts. float16 GPUs miss this by orders of magnitude. */
const SMOKE_TOL = 1e-3;

/** A fixed 4x8 by 8x3 matmul checked against plain JavaScript. Returns null when it matches. */
export function smokeTest(): string | null {
  const a = Array.from({ length: 32 }, (_, i) => Math.sin(i * 0.7) * 3);
  const b = Array.from({ length: 24 }, (_, i) => Math.cos(i * 1.3) * 2);
  let got: Float32Array;
  try {
    got = tf.tidy(() => tf.matMul(tf.tensor2d(a, [4, 8]), tf.tensor2d(b, [8, 3])).dataSync() as Float32Array);
  } catch (err) {
    return `smoke test threw: ${err instanceof Error ? err.message : String(err)}`;
  }
  let worst = 0;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 3; c++) {
      let want = 0;
      for (let k = 0; k < 8; k++) want += a[r * 8 + k]! * b[k * 3 + c]!;
      worst = Math.max(worst, Math.abs(want - (got[r * 3 + c] ?? NaN)));
    }
  }
  if (!(worst <= SMOKE_TOL)) return `smoke test off by ${worst}`;
  return null;
}

async function tryBackend(name: BackendName): Promise<string | null> {
  if (!tf.findBackendFactory(name)) {
    return name === 'webgpu' ? 'not registered (@tensorflow/tfjs-backend-webgpu is not installed)' : 'not registered';
  }
  let ok = false;
  try {
    ok = await tf.setBackend(name);
    if (ok) await tf.ready();
  } catch (err) {
    return `failed to initialise: ${err instanceof Error ? err.message : String(err)}`;
  }
  if (!ok || tf.getBackend() !== name) return 'failed to initialise';
  return smokeTest();
}

/**
 * Switch TF.js to the first backend in `order` that initialises and passes the smoke test. CPU is
 * always tried last, even when `order` leaves it out, so this never leaves TF.js without a backend.
 */
export async function selectBackend(order: BackendName[] = DEFAULT_ORDER): Promise<BackendChoice> {
  const list = order.includes('cpu') ? order : [...order, 'cpu' as const];
  const attempts: BackendAttempt[] = [];
  for (const name of list) {
    const reason = await tryBackend(name);
    attempts.push(reason ? { name, ok: false, reason } : { name, ok: true });
    if (!reason) return { active: tf.getBackend(), attempts, fellBack: name !== list[0] };
  }
  throw new Error(`no TF.js backend worked: ${attempts.map((a) => `${a.name} ${a.reason}`).join('; ')}`);
}
