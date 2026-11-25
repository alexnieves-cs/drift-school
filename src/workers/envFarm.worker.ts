/** Browser entry for one env-farm shard. All of the logic lives in envFarmShard.ts. */
import { serveShard } from './envFarmShard';

interface ShardScope {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage(data: unknown): void;
  close(): void;
}

const scope = globalThis as unknown as ShardScope;

serveShard({
  post: (msg) => scope.postMessage(msg),
  onMessage: (fn) => {
    scope.onmessage = (ev) => fn(ev.data);
  },
  close: () => scope.close(),
});
