import type { EpisodeConfig, TrackSpec } from '../shared/types';
import type { RNG } from '../shared/rng';
import { makeRng } from '../shared/rng';
import { generateTrack } from '../track/generate';
import { defaultCarParams } from '../sim/car/params';
import { SURFACES, latencyFor, noiseFor, sampleCar, type DRLevel, type Surface } from '../sim/dr';

/**
 * Training scenarios: which track, surface and domain-randomization level an episode gets.
 *
 * SPEC curriculum: stage 0 is a wide oval at high grip with no DR; stage 1 gentle loops;
 * stage 2 tighter corners with mild (low) DR; stage 3 hairpins/chicanes with full (high) DR
 * and mixed surfaces. Stage 4 (racing, low-grip) is not trained by the recipe.
 */

export type SurfaceChoice = 'native' | Surface['name'];

export interface StageProfile {
  /** Probabilities for DR off / low / high. */
  dr: [number, number, number];
  /** Probabilities for native track mu, then uniform asphalt, carpet, dirt. */
  surface: [number, number, number, number];
}

export const STAGE_PROFILES: StageProfile[] = [
  { dr: [1, 0, 0], surface: [1, 0, 0, 0] },
  { dr: [0.6, 0.4, 0], surface: [0.7, 0.1, 0.1, 0.1] },
  { dr: [0.3, 0.5, 0.2], surface: [0.4, 0.2, 0.2, 0.2] },
  { dr: [0.1, 0.4, 0.5], surface: [0.4, 0.2, 0.2, 0.2] },
];

const DR_LEVELS: DRLevel[] = ['off', 'low', 'high'];
const SURFACE_CHOICES: SurfaceChoice[] = ['native', 'asphalt', 'carpet', 'dirt'];

function pick<T>(rng: RNG, items: T[], probs: number[]): T {
  let u = rng.next() * probs.reduce((a, b) => a + b, 0);
  for (let i = 0; i < items.length; i++) {
    u -= probs[i]!;
    if (u <= 0) return items[i]!;
  }
  return items[items.length - 1]!;
}

/** Same track, uniform surface mu. Cached per (track, surface) because VecEnv builds rays from it. */
export function surfaceTrack(track: TrackSpec, surface: SurfaceChoice): TrackSpec {
  if (surface === 'native') return track;
  const mu = SURFACES[surface];
  return {
    ...track,
    id: `${track.id}@${surface}`,
    samples: track.samples.map((s) => ({ ...s, mu })),
  };
}

export interface Scenario {
  track: TrackSpec;
  stage: number;
  dr: DRLevel;
  surface: SurfaceChoice;
}

/** Track pools per stage, generated from a fixed seed list, plus a seeded scenario sampler. */
export class ScenarioSet {
  readonly seeds: number[];
  private readonly pools = new Map<number, TrackSpec[]>();
  private readonly surfaced = new Map<string, TrackSpec>();

  constructor(seeds: number[]) {
    this.seeds = seeds.slice();
  }

  tracks(stage: number): TrackSpec[] {
    let pool = this.pools.get(stage);
    if (!pool) {
      pool = this.seeds.map((seed) => generateTrack({ seed, stage }));
      this.pools.set(stage, pool);
    }
    return pool;
  }

  private withSurface(track: TrackSpec, surface: SurfaceChoice): TrackSpec {
    if (surface === 'native') return track;
    const key = `${track.id}@${surface}`;
    let t = this.surfaced.get(key);
    if (!t) {
      t = surfaceTrack(track, surface);
      this.surfaced.set(key, t);
    }
    return t;
  }

  /**
   * `maxStage` is the highest stage in play. 80% of episodes use `maxStage`,
   * 20% replay a uniformly chosen earlier stage (SPEC: 20% replay of prior stages).
   */
  sample(rng: RNG, maxStage: number, force?: { dr?: DRLevel }): Scenario {
    let stage = maxStage;
    if (maxStage > 0 && rng.next() < 0.2) stage = rng.int(maxStage);
    const profile = STAGE_PROFILES[Math.min(stage, STAGE_PROFILES.length - 1)]!;
    const dr = force?.dr ?? pick(rng, DR_LEVELS, profile.dr);
    const surface = pick(rng, SURFACE_CHOICES, profile.surface);
    const pool = this.tracks(stage);
    const track = this.withSurface(pool[rng.int(pool.length)]!, surface);
    return { track, stage, dr, surface };
  }
}

/** Episode config for a scenario. Deterministic in (seed, track, index). */
export function episodeConfig(
  sc: { track: TrackSpec; stage: number; dr: DRLevel },
  seed: number,
  index: number,
  opts: { grip?: number; massScale?: number; lapsTarget?: number } = {},
): EpisodeConfig {
  const rng = makeRng(seed * 10007 + index * 13 + (sc.track.seed ?? 0), `dr/${sc.track.id}`);
  let car = sampleCar(defaultCarParams(), rng, sc.dr, opts.grip ?? 1);
  const m = opts.massScale ?? 1;
  if (m !== 1) {
    car = { ...car, mass: car.mass * m };
    car.yawInertia *= m;
  }
  return {
    stage: sc.stage,
    car,
    noiseScale: noiseFor(rng, sc.dr),
    latencyTicks: latencyFor(rng, sc.dr),
    trackId: sc.track.id,
    lapsTarget: opts.lapsTarget ?? (sc.stage <= 1 ? 1 : 2),
    seed: (sc.track.seed ?? 0) + index + seed,
  };
}
