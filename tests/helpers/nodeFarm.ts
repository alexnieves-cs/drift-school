/**
 * Runs env-farm shards in Node worker threads, for tests and scripts/farm-bench.ts. The shard entry
 * is bundled with esbuild once per process, the same way tests/shared/worker-ping.test.ts does it.
 */
import esbuild from 'esbuild';
import { Worker } from 'node:worker_threads';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import type { FarmEndpoint, SpawnShard } from '../../src/workers/envFarm';

let bundled: string | null = null;

async function bundle(): Promise<string> {
  if (bundled) return bundled;
  const built = await esbuild.build({
    entryPoints: [resolve(__dirname, 'farmNodeWorker.ts')],
    bundle: true,
    format: 'esm',
    platform: 'node',
    write: false,
  });
  const dir = mkdtempSync(join(tmpdir(), 'ds-farm-'));
  const file = join(dir, 'farm-shard.mjs');
  writeFileSync(file, built.outputFiles[0]!.text);
  bundled = file;
  return file;
}

export async function nodeSpawn(): Promise<SpawnShard> {
  const file = await bundle();
  return (): FarmEndpoint => {
    const worker = new Worker(file);
    return {
      post: (msg) => worker.postMessage(msg),
      onMessage: (fn) => worker.on('message', fn),
      terminate: () => worker.terminate(),
    };
  };
}
