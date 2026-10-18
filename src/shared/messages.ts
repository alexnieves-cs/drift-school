import type { CameraMode, EpisodeConfig, GenStat, TrackSpec } from './types';

export type FineTuneOpts = {
  entropy: number;
  entropyUpdates: number;
  lr: number;
};

export type InMessage =
  | { type: 'ping'; t: number }
  | { type: 'init'; n: number; tracks: TrackSpec[]; cfgs: EpisodeConfig[] }
  | { type: 'reset' }
  | { type: 'step'; actions?: Float32Array }
  | { type: 'weights'; weights: ArrayBuffer[]; gen: number }
  | {
      type: 'setTrackPool';
      tracks: TrackSpec[];
      mix: number[];
      fineTune?: boolean;
      forkName?: string;
    }
  | { type: 'replayReq'; gen: number }
  | { type: 'checkpointSave'; name: string }
  | { type: 'checkpointLoad'; name: string }
  | { type: 'setDR'; grip: number; massScale: number; level: 'low' | 'high' | 'off' }
  | { type: 'setStage'; stage: number }
  | { type: 'setCamera'; mode: CameraMode }
  | { type: 'pause'; paused: boolean }
  | { type: 'startTrain' };

export type OutMessage =
  | { type: 'pong'; t: number }
  | { type: 'ready' }
  | { type: 'metrics'; stat: GenStat }
  | { type: 'ghosts'; poses: Float32Array; n: number; gen: number; trackId: string }
  | { type: 'replayRes'; gen: number; poses: Float32Array; n: number }
  | { type: 'trackAck'; trackId: string; envCount: number }
  | { type: 'weightHash'; hash: string }
  | { type: 'checkpointSaved'; name: string }
  | { type: 'checkpointLoaded'; name: string; ok: boolean }
  | { type: 'error'; message: string };

export function handlePing(msg: { type: string; t?: number }): { type: 'pong'; t: number } | null {
  if (msg.type === 'ping') return { type: 'pong', t: msg.t ?? 0 };
  return null;
}
