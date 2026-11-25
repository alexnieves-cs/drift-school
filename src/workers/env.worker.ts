import { VecEnv } from '../sim/vecEnv';
import { generateTrack } from '../track/generate';
import { buildTrackFromPoints } from '../track/build';
import { defaultCarParams } from '../sim/car/params';
import { handlePing, parseEnvIn, stamp, type EnvOut } from '../shared/messages';
import type { EpisodeConfig, TrackSpec } from '../shared/types';

interface WorkerScope {
  onmessage: ((ev: MessageEvent) => void) | null;
  postMessage(data: unknown): void;
}

const scope = globalThis as unknown as WorkerScope;
let env: VecEnv | null = null;
let tracks: TrackSpec[] = [];

function post(data: EnvOut): void {
  scope.postMessage(stamp(data));
}

function cfgs(list: TrackSpec[]): EpisodeConfig[] {
  return list.map((track) => ({
    stage: 0,
    car: defaultCarParams(),
    noiseScale: 0,
    latencyTicks: 0,
    trackId: track.id,
  }));
}

scope.onmessage = (ev: MessageEvent) => {
  const parsed = parseEnvIn(ev.data);
  if (!parsed.ok) {
    post({ type: 'error', message: `rejected message: ${parsed.error}` });
    return;
  }
  const msg = parsed.msg;
  const pong = handlePing(msg);
  if (pong) {
    post(pong);
    return;
  }
  if (msg.type === 'init') {
    const n = msg.n;
    const seeds = msg.seeds.length ? msg.seeds : [1, 2, 3, 4];
    tracks = seeds.map((seed) => generateTrack({ seed, stage: msg.stage }));
    env = new VecEnv(n);
    const pool = Array.from({ length: n }, (_, i) => tracks[i % tracks.length]!);
    env.reset(pool, cfgs(pool));
    post({ type: 'ready', trackId: env.trackId(0) });
    return;
  }
  if (!env) return;
  if (msg.type === 'step') {
    const actions = msg.actions ?? env.baselineActions();
    env.step(actions);
    post({ type: 'ghosts', poses: new Float32Array(env.poses()), n: env.n, gen: 0, trackId: env.trackId(0), ids: env.trackIds() });
    return;
  }
  if (msg.type === 'setTrackPool') {
    const points = Float32Array.from(msg.points);
    const user = buildTrackFromPoints(points, { halfWidth: msg.halfWidth, mu: msg.mu, source: 'user' });
    const pool = Array.from({ length: env.n }, (_, i) => (i % 5 === 0 ? tracks[i % Math.max(1, tracks.length)]! : user));
    env.reset(pool, cfgs(pool));
    post({ type: 'trackAck', trackId: user.id, envCount: env.trackIds().filter((id) => id === user.id).length });
  }
};
