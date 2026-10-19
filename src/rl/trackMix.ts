import type { TrackSpec } from '../shared/types';
import type { RNG } from '../shared/rng';

export interface MixState {
  entropy: number;
  lr: number;
  fineTuneUpdatesLeft: number;
}

/**
 * Weighted track pool. Fine-tune mode is 70% user track / 30% procedural,
 * with an entropy bump to 0.02 for 20 updates and an LR warm restart at 3e-4.
 */
export class TrackMix {
  tracks: TrackSpec[] = [];
  weights: number[] = [];
  entropy = 0.01;
  lr = 3e-4;
  fineTuneUpdatesLeft = 0;
  userId: string | null = null;

  setPool(tracks: TrackSpec[], weights?: number[]): void {
    this.tracks = tracks.slice();
    if (!tracks.length) {
      this.weights = [];
      return;
    }
    if (weights && weights.length === tracks.length) {
      const s = weights.reduce((a, b) => a + b, 0) || 1;
      this.weights = weights.map((w) => w / s);
    } else {
      this.weights = tracks.map(() => 1 / tracks.length);
    }
  }

  /** 70/30 user/procedural. `user` is placed first. */
  startFineTune(user: TrackSpec, procedural: TrackSpec[]): void {
    const tracks = [user, ...procedural];
    const weights = tracks.map((_, i) => (i === 0 ? 0.7 : 0.3 / Math.max(1, procedural.length)));
    this.setPool(tracks, weights);
    this.userId = user.id;
    this.entropy = 0.02;
    this.lr = 3e-4;
    this.fineTuneUpdatesLeft = 20;
  }

  onUpdate(): void {
    if (this.fineTuneUpdatesLeft > 0) {
      this.fineTuneUpdatesLeft--;
      if (this.fineTuneUpdatesLeft === 0) this.entropy = 0.01;
    }
  }

  sample(rng: RNG): TrackSpec {
    if (!this.tracks.length) throw new Error('empty track pool');
    let u = rng.next();
    for (let i = 0; i < this.tracks.length; i++) {
      u -= this.weights[i] ?? 0;
      if (u <= 0) return this.tracks[i]!;
    }
    return this.tracks[this.tracks.length - 1]!;
  }

  state(): MixState {
    return { entropy: this.entropy, lr: this.lr, fineTuneUpdatesLeft: this.fineTuneUpdatesLeft };
  }
}
