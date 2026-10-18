import { parentPort } from 'node:worker_threads';
import { handlePing } from '../shared/messages';

parentPort?.on('message', (msg: { type: string; t?: number }) => {
  const out = handlePing(msg);
  if (out) parentPort?.postMessage(out);
});
