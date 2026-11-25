/** Node worker_threads entry for one env-farm shard, bundled by tests/helpers/nodeFarm.ts. */
import { parentPort } from 'node:worker_threads';
import { serveShard } from '../../src/workers/envFarmShard';

const port = parentPort!;
serveShard({
  post: (msg) => port.postMessage(msg),
  onMessage: (fn) => port.on('message', fn),
  close: () => port.close(),
});
