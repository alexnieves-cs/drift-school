import type { StepInfo } from '../shared/types';

export interface RewardInput {
  ds: number;
  dt: number;
  lapDone: boolean;
  lapTime: number;
  tRef: number;
  offTrack: boolean;
  wallHit: boolean;
  impact: number;
  inContact: boolean;
  steer: number;
  throttle: number;
  prevSteer: number;
  prevThrottle: number;
  speed: number;
  wrongWay: boolean;
  /** Body slip angle in radians, for the drift bonus. */
  slip?: number;
  /** Drift bonus weight; 0 or missing leaves the reward exactly as before. */
  slipWeight?: number;
  /** Slip angle in radians where the bonus stops growing. Default DRIFT_TARGET_DEFAULT. */
  slipTarget?: number;
}

export interface RewardOut {
  reward: number;
  done: boolean;
  truncated: boolean;
  info: StepInfo;
}

export class EpisodeBook {
  offTime = 0;
  wrongTime = 0;
  stuckTime = 0;
  prevSteer = 0;
  prevThrottle = 0;
  steps = 0;
  crashed = false;

  reset(): void {
    this.offTime = 0;
    this.wrongTime = 0;
    this.stuckTime = 0;
    this.prevSteer = 0;
    this.prevThrottle = 0;
    this.steps = 0;
    this.crashed = false;
  }
}

export const MAX_DS = 0.55;

/** Below this speed (m/s) a large slip angle is a spin or a crawl, not a drift. */
export const DRIFT_MIN_SPEED = 1.5;
export const DRIFT_TARGET_DEFAULT = (25 * Math.PI) / 180;

/**
 * Drift bonus: `slipWeight * min(|slip|, target) / target * ds`. It scales with forward progress,
 * so sliding without getting anywhere earns nothing, and a slide at the target angle earns
 * (1 + slipWeight) times the progress reward. Nothing is paid off track, against a wall, going the
 * wrong way or below DRIFT_MIN_SPEED.
 */
export function slipBonus(inp: RewardInput): number {
  const w = inp.slipWeight ?? 0;
  if (!(w > 0)) return 0;
  if (!(inp.ds > 0 && inp.ds < MAX_DS)) return 0;
  if (inp.speed <= DRIFT_MIN_SPEED || inp.offTrack || inp.wallHit || inp.inContact || inp.wrongWay) return 0;
  const target = inp.slipTarget && inp.slipTarget > 0 ? inp.slipTarget : DRIFT_TARGET_DEFAULT;
  return (w * Math.min(Math.abs(inp.slip ?? 0), target) * inp.ds) / target;
}

export function scoreStep(inp: RewardInput, book: EpisodeBook, stepCap: number, lapsTarget: number, laps: number): RewardOut {
  let reward = -0.01;
  if (inp.ds > 0 && inp.ds < MAX_DS) reward += inp.ds;
  if (inp.slipWeight && inp.slipWeight > 0) reward += slipBonus(inp);
  if (inp.lapDone) {
    reward += 10;
    if (inp.lapTime > 0.5 && inp.tRef > 0 && inp.lapTime < inp.tRef) {
      reward += 20 * (inp.tRef / inp.lapTime - 1);
    }
  }
  if (inp.offTrack) {
    reward -= 0.5;
    book.offTime += inp.dt;
  } else book.offTime = 0;
  if (inp.wallHit) {
    reward -= 5 * Math.min(2, Math.max(0.2, inp.impact));
    book.crashed = true;
  }
  if (inp.inContact) reward -= 0.2;
  const dSteer = inp.steer - inp.prevSteer;
  const dThr = inp.throttle - inp.prevThrottle;
  reward -= 0.02 * (dSteer * dSteer + dThr * dThr);
  if (inp.speed > 3) reward -= 0.005 * inp.steer * inp.steer;
  if (inp.wrongWay) {
    reward -= 1;
    book.wrongTime += inp.dt;
  } else book.wrongTime = 0;
  if (inp.speed < 0.2) book.stuckTime += inp.dt;
  else book.stuckTime = 0;
  book.steps++;
  book.prevSteer = inp.steer;
  book.prevThrottle = inp.throttle;

  let done = false;
  let truncated = false;
  if (laps >= lapsTarget) done = true;
  if (book.offTime > 1.5) done = true;
  if (book.wrongTime > 2) done = true;
  if (book.stuckTime > 3) done = true;
  if (book.steps >= stepCap) truncated = true;

  return {
    reward,
    done,
    truncated,
    info: {
      progress: 0,
      lapDone: inp.lapDone,
      lapTime: inp.lapDone ? inp.lapTime : undefined,
      offTrack: inp.offTrack,
      wallHit: inp.wallHit || inp.inContact,
      speed: inp.speed,
      wrongWay: inp.wrongWay,
      stuck: book.stuckTime > 3,
      rewardRaw: reward,
    },
  };
}
