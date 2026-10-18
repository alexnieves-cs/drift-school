/**
 * Ghost replay of a stored pose tape. The tape is a flat array of 7-float poses
 * (x, y, z, qw, qx, qy, qz) recorded at TAPE_HZ. Playback is a pure function of the tape and
 * the elapsed time, so the same row always replays the same path.
 */
import { POSE_STRIDE } from './interp';

/** The trainer records one tape frame every second control step (50 Hz control). */
export const TAPE_HZ = 25;

export interface ReplayPose {
  x: number;
  y: number;
  z: number;
  /** Sim-convention yaw (radians), the same sign as HeroState.yaw. */
  yaw: number;
}

export function tapeFrames(tape: ArrayLike<number>): number {
  return Math.floor(tape.length / POSE_STRIDE);
}

/** Pose at tape time `t` seconds. Linear in position, shortest-arc in yaw. Clamped to the tape. */
export function poseAt(tape: ArrayLike<number>, t: number, out: ReplayPose = { x: 0, y: 0, z: 0, yaw: 0 }): ReplayPose {
  const frames = tapeFrames(tape);
  if (frames === 0) return out;
  const f = Math.min(frames - 1, Math.max(0, t * TAPE_HZ));
  const i = Math.floor(f);
  const j = Math.min(frames - 1, i + 1);
  const a = f - i;
  const o = i * POSE_STRIDE;
  const p = j * POSE_STRIDE;
  out.x = tape[o]! + (tape[p]! - tape[o]!) * a;
  out.y = tape[o + 1]! + (tape[p + 1]! - tape[o + 1]!) * a;
  out.z = tape[o + 2]! + (tape[p + 2]! - tape[o + 2]!) * a;
  const y0 = 2 * Math.atan2(tape[o + 5]!, tape[o + 3]!);
  const y1 = 2 * Math.atan2(tape[p + 5]!, tape[p + 3]!);
  let d = y1 - y0;
  d -= Math.round(d / (2 * Math.PI)) * 2 * Math.PI;
  out.yaw = y0 + d * a;
  return out;
}

export class ReplayPlayer {
  private tape: ArrayLike<number> | null = null;
  private time = 0;
  private gen_ = -1;
  private readonly pose: ReplayPose = { x: 0, y: 0, z: 0, yaw: 0 };

  get active(): boolean {
    return this.tape !== null;
  }

  get gen(): number {
    return this.gen_;
  }

  get duration(): number {
    return this.tape ? Math.max(0, (tapeFrames(this.tape) - 1) / TAPE_HZ) : 0;
  }

  start(gen: number, tape: ArrayLike<number>): boolean {
    if (tapeFrames(tape) < 2) return false;
    this.tape = tape;
    this.gen_ = gen;
    this.time = 0;
    return true;
  }

  stop(): void {
    this.tape = null;
    this.gen_ = -1;
  }

  /** Advance by `dt` seconds. Returns the pose to draw, or null when the replay has finished. */
  advance(dt: number): ReplayPose | null {
    if (!this.tape) return null;
    this.time += dt;
    if (this.time > this.duration + 0.4) {
      this.stop();
      return null;
    }
    return poseAt(this.tape, this.time, this.pose);
  }
}
