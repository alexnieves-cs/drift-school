import { ACT_DIM, OBS_DIM, type EpisodeConfig, type StepInfo, type TrackSpec } from '../shared/types';
import { CONTROL_DT } from '../shared/constants';
import { makeRng, type RNG } from '../shared/rng';
import { nearestSample } from '../track/sample';
import { purePursuit } from '../track/baseline';
import { Vehicle } from './car/vehicle';
import { defaultCarParams } from './car/params';
import { RayField, fillObservation, wheelsOff } from './sensors';
import { EpisodeBook, MAX_DS as MAX_PLAUSIBLE_DS, scoreStep } from './reward';

interface Slot {
  car: Vehicle;
  track: TrackSpec;
  cfg: EpisodeConfig;
  rays: RayField;
  book: EpisodeBook;
  hint: number;
  prevS: number;
  lap: number;
  lapTime: number;
  /** Time of the last completed lap this episode. */
  lastLapTime: number;
  /** Forward arc length since the last start-line crossing. A lap needs ~a full circuit of it. */
  lapProgress: number;
  stepInLap: number;
  rng: RNG;
  delay: number[];
  action: [number, number];
  contact: boolean;
  returning: number;
}

function yawOf(tx: number, tz: number): number {
  return Math.atan2(tz, tx);
}

/** Step cap for an episode. The `time_left` observation counts down the same budget. */
export function stepCapFor(trackLength: number, lapsTarget: number): number {
  return Math.max(200, Math.round(((trackLength / 1.5) * lapsTarget * 1.8) / CONTROL_DT));
}

/** Roll angle (rad) that ends an episode as a flip. The vehicle model clamps roll to +/-0.9 rad, so SPEC's 80 degrees is unreachable. */
const FLIP_ROLL = 0.85;

/** Fresh track and episode config for a slot that just ended an episode. */
export type EpisodeSampler = (slot: number) => { track: TrackSpec; cfg: EpisodeConfig };

export interface CarSnap {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  speed: number;
  steer: number;
  throttle: number;
  slip: number[];
  load: number[];
  lap: number;
  lapTime: number;
  s: number;
  yawRate: number;
  vx: number;
  vz: number;
  surfaceMu: number;
}

export class VecEnv {
  readonly n: number;
  readonly obs: Float32Array;
  readonly rew: Float32Array;
  readonly done: Uint8Array;
  readonly trunc: Uint8Array;
  readonly poseBuf: Float32Array;
  readonly infos: StepInfo[];
  private slots: Slot[] = [];
  /** When set, every slot draws a new track and car config at each episode end (per-episode DR and track mixing). */
  sampler: EpisodeSampler | null = null;
  /** Episodes finished since the last consume. */
  readonly finished: { lap: boolean; lapTime?: number; crash: boolean; ret: number; trackId: string; tRef: number; speed: number }[] = [];

  constructor(n: number) {
    this.n = n;
    this.obs = new Float32Array(n * OBS_DIM);
    this.rew = new Float32Array(n);
    this.done = new Uint8Array(n);
    this.trunc = new Uint8Array(n);
    this.poseBuf = new Float32Array(n * 7);
    this.infos = Array.from({ length: n }, () => ({
      progress: 0,
      lapDone: false,
      offTrack: false,
      wallHit: false,
      speed: 0,
      wrongWay: false,
      stuck: false,
      rewardRaw: 0,
    }));
  }

  reset(tracks: TrackSpec[], cfgs: EpisodeConfig[]): Float32Array {
    this.slots = [];
    this.finished.length = 0;
    for (let i = 0; i < this.n; i++) {
      const track = tracks[i % tracks.length]!;
      const cfg = cfgs[i % cfgs.length] ?? {
        stage: 0,
        car: defaultCarParams(),
        noiseScale: 0,
        latencyTicks: 0,
        trackId: track.id,
      };
      const car = new Vehicle(cfg.car);
      const slot: Slot = {
        car,
        track,
        cfg,
        rays: new RayField(),
        book: new EpisodeBook(),
        hint: 0,
        prevS: 0,
        lap: 0,
        lapTime: 0,
        lastLapTime: 0,
        lapProgress: 0,
        stepInLap: 0,
        rng: makeRng(cfg.seed ?? i + 1, `env/${i}/${track.id}`),
        delay: [],
        action: [0, 0.4],
        contact: false,
        returning: 0,
      };
      slot.rays.setTrack(track);
      this.place(slot);
      this.slots.push(slot);
      this.writeObs(i);
    }
    return this.obs;
  }

  private place(slot: Slot): void {
    const s0 = slot.track.samples[slot.track.startIndex] ?? slot.track.samples[0];
    if (!s0) return;
    slot.car.reset({ x: s0.x, z: s0.z, yaw: yawOf(s0.tx, s0.tz), speed: 0.4 });
    slot.book.reset();
    slot.hint = slot.track.startIndex;
    slot.prevS = s0.s;
    slot.lap = 0;
    slot.lapTime = 0;
    slot.lastLapTime = 0;
    slot.lapProgress = s0.s;
    slot.stepInLap = 0;
    slot.contact = false;
    slot.returning = 0;
    slot.action = [0, 0.4];
    slot.delay = [];
  }

  step(actions: Float32Array): { obs: Float32Array; rew: Float32Array; done: Uint8Array; trunc: Uint8Array; infos: StepInfo[] } {
    for (let i = 0; i < this.n; i++) {
      const slot = this.slots[i];
      if (!slot) continue;
      let steer = actions[i * ACT_DIM] ?? 0;
      let throttle = actions[i * ACT_DIM + 1] ?? 0;
      if (slot.cfg.latencyTicks > 0) {
        slot.delay.push(steer, throttle);
        const lag = slot.cfg.latencyTicks * 2;
        if (slot.delay.length > lag) {
          steer = slot.delay.shift()!;
          throttle = slot.delay.shift()!;
        }
      }
      const pre = slot.track.samples[slot.hint];
      slot.car.surfaceMu = pre && pre.mu > 0 ? pre.mu / 1.05 : 1;
      const st = slot.car.step(steer, throttle, CONTROL_DT);
      const q = nearestSample(slot.track, slot.car.px, slot.car.pz, slot.hint);
      slot.hint = q.index;
      let ds = q.sample.s - slot.prevS;
      const half = slot.track.length * 0.5;
      let crossedForward = false;
      if (ds < -half) {
        ds += slot.track.length;
        crossedForward = true;
      } else if (ds > half) {
        // Backwards across the start line.
        ds -= slot.track.length;
      }
      if (ds < 0) ds = Math.max(ds, -0.05);
      if (ds < MAX_PLAUSIBLE_DS) slot.lapProgress += ds;
      // Crossing the line only counts after a near-complete circuit, so reversing over it and
      // driving forward again cannot score a lap.
      const lapDone = crossedForward && slot.lapProgress >= 0.9 * slot.track.length;
      if (crossedForward) slot.lapProgress = 0;
      slot.lapTime += CONTROL_DT;
      if (lapDone) {
        slot.lap++;
        slot.car.lap = slot.lap;
      }
      slot.car.s = q.sample.s;
      slot.car.lapTime = slot.lapTime;
      const off = wheelsOff(slot.track, slot.car, slot.hint) >= 2;
      const cy = Math.cos(slot.car.yaw);
      const sy = Math.sin(slot.car.yaw);
      const align = cy * q.sample.tx + sy * q.sample.tz;
      const wrong = align < -0.5;
      const wall = this.wallPush(slot);
      const lapsTarget = slot.cfg.lapsTarget ?? (slot.cfg.stage <= 1 ? 1 : 2);
      const stepCap = stepCapFor(slot.track.length, lapsTarget);
      const scored = scoreStep(
        {
          ds,
          dt: CONTROL_DT,
          lapDone,
          lapTime: slot.lapTime,
          tRef: slot.track.tRef,
          offTrack: off,
          wallHit: wall.hit,
          impact: wall.impact,
          inContact: slot.contact,
          steer,
          throttle,
          prevSteer: slot.book.prevSteer,
          prevThrottle: slot.book.prevThrottle,
          speed: slot.car.speed(),
          wrongWay: wrong,
        },
        slot.book,
        stepCap,
        lapsTarget,
        slot.lap,
      );
      if (Math.abs(slot.car.roll) > FLIP_ROLL) scored.done = true;
      slot.prevS = q.sample.s;
      slot.returning += scored.reward;
      this.rew[i] = scored.reward;
      this.done[i] = scored.done ? 1 : 0;
      this.trunc[i] = scored.truncated ? 1 : 0;
      const info = this.infos[i]!;
      info.progress = q.sample.s;
      info.lapDone = lapDone;
      info.lapTime = lapDone ? slot.lapTime : undefined;
      if (lapDone) slot.lastLapTime = slot.lapTime;
      info.offTrack = off;
      info.wallHit = wall.hit || slot.contact;
      info.speed = slot.car.speed();
      info.wrongWay = wrong;
      info.stuck = scored.info.stuck;
      info.rewardRaw = scored.reward;
      slot.action = [steer, throttle];
      if (scored.done || scored.truncated) {
        this.finished.push({
          lap: slot.lap >= 1,
          lapTime: slot.lap >= 1 ? slot.lastLapTime : undefined,
          crash: slot.book.crashed || off,
          ret: slot.returning,
          trackId: slot.track.id,
          tRef: slot.track.tRef,
          speed: info.speed,
        });
        this.startNext(i, slot);
      } else if (lapDone) {
        slot.lastLapTime = slot.lapTime;
        slot.lapTime = 0;
      }
      this.writeObs(i);
      void st;
    }
    return { obs: this.obs, rew: this.rew, done: this.done, trunc: this.trunc, infos: this.infos };
  }

  private startNext(i: number, slot: Slot): void {
    if (this.sampler) {
      const next = this.sampler(i);
      if (next.track !== slot.track) {
        slot.track = next.track;
        slot.rays.setTrack(next.track);
      }
      slot.cfg = next.cfg;
      slot.car = new Vehicle(next.cfg.car);
    }
    this.place(slot);
  }

  /** Cheap polyline resolution. Same wall geometry Rapier uses for the driven car. */
  private wallPush(slot: Slot): { hit: boolean; impact: number } {
    const margin = 0.18;
    let hit = false;
    let impact = 0;
    const near = slot.rays.gather(slot.car.px, slot.car.pz, margin + 0.5);
    for (let k = 0; k < slot.rays.nearCount; k++) {
      const seg = slot.rays.segs[near[k]!]!;
      const x1 = seg.x1;
      const z1 = seg.z1;
      const dx = seg.x2 - x1;
      const dz = seg.z2 - z1;
      const len2 = dx * dx + dz * dz || 1;
      let t = ((slot.car.px - x1) * dx + (slot.car.pz - z1) * dz) / len2;
      t = Math.max(0, Math.min(1, t));
      const cx = x1 + dx * t;
      const cz = z1 + dz * t;
      const ox = slot.car.px - cx;
      const oz = slot.car.pz - cz;
      const dist = Math.hypot(ox, oz);
      if (dist < margin) {
        hit = true;
        const nx = ox / (dist || 1);
        const nz = oz / (dist || 1);
        slot.car.px += nx * (margin - dist);
        slot.car.pz += nz * (margin - dist);
        const cy = Math.cos(slot.car.yaw);
        const sy = Math.sin(slot.car.yaw);
        const worldVx = cy * slot.car.vx - sy * slot.car.vz;
        const worldVz = sy * slot.car.vx + cy * slot.car.vz;
        const vn = worldVx * nx + worldVz * nz;
        if (vn < 0) {
          const e = 0.3;
          const wx = worldVx - (1 + e) * vn * nx;
          const wz = worldVz - (1 + e) * vn * nz;
          slot.car.vx = cy * wx + sy * wz;
          slot.car.vz = -sy * wx + cy * wz;
          impact = Math.max(impact, -vn);
        }
      }
    }
    slot.contact = hit;
    return { hit: impact > 0.4, impact };
  }

  private writeObs(i: number): void {
    const slot = this.slots[i]!;
    const cap = stepCapFor(slot.track.length, slot.cfg.lapsTarget ?? (slot.cfg.stage <= 1 ? 1 : 2));
    const timeLeft = Math.max(0, 1 - slot.book.steps / cap);
    slot.hint = fillObservation(
      this.obs,
      i * OBS_DIM,
      slot.track,
      slot.car,
      slot.rays,
      slot.hint,
      slot.action[0],
      slot.action[1],
      timeLeft,
      !!slot.cfg.raysOnly,
      slot.cfg.noiseScale > 0 ? slot.rng : null,
      slot.cfg.noiseScale,
    );
  }

  baselineActions(): Float32Array {
    const out = new Float32Array(this.n * ACT_DIM);
    for (let i = 0; i < this.n; i++) {
      const slot = this.slots[i];
      if (!slot) continue;
      const cmd = purePursuit(slot.car, slot.track, slot.hint);
      slot.hint = cmd.hint;
      out[i * ACT_DIM] = cmd.steer;
      out[i * ACT_DIM + 1] = cmd.throttle;
    }
    return out;
  }

  poses(): Float32Array {
    for (let i = 0; i < this.n; i++) {
      const car = this.slots[i]!.car;
      const cy = Math.cos(car.yaw / 2);
      const sy = Math.sin(car.yaw / 2);
      this.poseBuf[i * 7] = car.px;
      this.poseBuf[i * 7 + 1] = car.py;
      this.poseBuf[i * 7 + 2] = car.pz;
      this.poseBuf[i * 7 + 3] = cy;
      this.poseBuf[i * 7 + 4] = 0;
      this.poseBuf[i * 7 + 5] = sy;
      this.poseBuf[i * 7 + 6] = 0;
    }
    return this.poseBuf;
  }

  trackId(i = 0): string {
    return this.slots[i]?.track.id ?? '';
  }

  trackIds(): string[] {
    return this.slots.map((s) => s.track.id);
  }

  snapshot(i = 0): CarSnap | null {
    const slot = this.slots[i];
    if (!slot) return null;
    const car = slot.car;
    return {
      x: car.px,
      y: car.py,
      z: car.pz,
      yaw: car.yaw,
      pitch: car.pitch,
      roll: car.roll,
      speed: car.speed(),
      steer: car.steer,
      throttle: slot.action[1],
      slip: [car.slipA[0]!, car.slipA[1]!, car.slipA[2]!, car.slipA[3]!],
      load: [car.load[0]!, car.load[1]!, car.load[2]!, car.load[3]!],
      lap: slot.lap,
      lapTime: slot.lapTime,
      s: car.s,
      yawRate: car.wYaw,
      vx: car.vx,
      vz: car.vz,
      surfaceMu: car.surfaceMu,
    };
  }
}
