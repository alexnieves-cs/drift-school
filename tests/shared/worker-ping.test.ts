import { describe, expect, it } from 'vitest';
import esbuild from 'esbuild';
import { Worker } from 'node:worker_threads';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { handlePing } from '../../src/shared/messages';

describe('worker ping', () => {
  it('handlePing echoes the timestamp', () => {
    expect(handlePing({ type: 'ping', t: 123 })).toEqual({ type: 'pong', t: 123 });
    expect(handlePing({ type: 'step' })).toBeNull();
  });

  it('round-trips through a worker thread', async () => {
    const built = await esbuild.build({
      entryPoints: ['src/workers/ping.worker.ts'],
      bundle: true,
      format: 'esm',
      platform: 'node',
      write: false,
    });
    const dir = mkdtempSync(join(tmpdir(), 'ds-ping-'));
    const file = join(dir, 'ping.mjs');
    writeFileSync(file, built.outputFiles[0]!.text);
    const worker = new Worker(file);
    const pong = await new Promise<{ type: string; t: number }>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('ping timeout')), 5000);
      worker.once('message', (msg) => {
        clearTimeout(timer);
        resolve(msg);
      });
      worker.postMessage({ type: 'ping', t: 99 });
    });
    await worker.terminate();
    expect(pong).toEqual({ type: 'pong', t: 99 });
  });
});
