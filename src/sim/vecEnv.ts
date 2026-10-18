import { ACT_DIM, OBS_DIM, type EpisodeConfig, type StepInfo, type TrackSpec } from '../shared/types';
import { CONTROL_DT } from '../shared/constants';
import { makeRng, type RNG } from '../shared/rng';
import { nearestSample } from '../track/sample';
import { newBaselineMemory, purePursuit, type BaselineController, type BaselineMemory, type BaselineOpts } from '../track/baseline';
import { Vehicle } from './car/vehicle';
import { defaultCarParams } from './car/params';
import { RayField, fillObservation, wheelsOff } from './sensors';
import { EpisodeBook, scoreStep } from './reward';
import { getWallGrid, resolveWallContact, type WallGrid, type WallResult } from './wallGrid';

interface Slot {
  car: Vehicle;
  track: TrackSpec;
  cfg: EpisodeConfig;
  rays: RayField;
  grid: WallGrid;
  book: EpisodeBook;
  hint: number;
  prevS: number;
  lap: number;
  lapTime: number;
  stepInLap: number;
  rng: RNG;
  delay: number[];
  /** Last applied [steer, throttle]; mutated in place so a step allocates nothing. */
  action: [number, number];
  contact: boolean;
  returning: number;
  mem: BaselineMemory;
}

function yawOf(tx: number, tz: number): number {
  return Math.atan2(tz, tx);
}

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
  private readonly wallOut: WallResult = { contact: false, hit: false, impact: 0 };
  /** Episodes finished since the last consume. */
  readonly finished: { lap: boolean; lapTime?: number; crash: boolean; ret: number; trackId: string; speed: number }[] = [];

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
        grid: getWallGrid(track.walls),
        book: new EpisodeBook(),
        hint: 0,
        prevS: 0,
        lap: 0,
        lapTime: 0,
        stepInLap: 0,
        rng: makeRng(cfg.seed ?? i + 1, `env/${i}/${track.id}`),
        delay: [],
        action: [0, 0.4],
        contact: false,
        returning: 0,
        mem: newBaselineMemory(),
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
    slot.stepInLap = 0;
    slot.contact = false;
    slot.returning = 0;
    slot.mem.pinned = 0;
    slot.mem.reversing = 0;
    slot.action[0] = 0;
    slot.action[1] = 0.4;
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
      let lapDone = false;
      if (ds < -slot.track.length * 0.5) {
        ds += slot.track.length;
        lapDone = true;
        slot.lap++;
        slot.car.lap = slot.lap;
      } else if (ds < 0) {
        ds = Math.max(ds, -0.05);
      }
      slot.lapTime += CONTROL_DT;
      slot.car.s = q.sample.s;
      slot.car.lapTime = slot.lapTime;
      const off = wheelsOff(slot.track, slot.car, slot.hint) >= 2;
      const cy = Math.cos(slot.car.yaw);
      const sy = Math.sin(slot.car.yaw);
      const align = cy * q.sample.tx + sy * q.sample.tz;
      const wrong = align < -0.5;
      const wall = this.wallPush(slot);
      const lapsTarget = slot.cfg.lapsTarget ?? (slot.cfg.stage <= 1 ? 1 : 2);
      const stepCap = Math.max(200, Math.round((slot.track.length / 1.5) * lapsTarget * 1.8 / CONTROL_DT));
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
      if (Math.abs(slot.car.roll) > (80 * Math.PI) / 180) scored.done = true;
      slot.prevS = q.sample.s;
      slot.returning += scored.reward;
      this.rew[i] = scored.reward;
      this.done[i] = scored.done ? 1 : 0;
      this.trunc[i] = scored.truncated ? 1 : 0;
      const info = this.infos[i]!;
      info.progress = q.sample.s;
      info.lapDone = lapDone;
      info.lapTime = lapDone ? slot.lapTime : undefined;
      info.offTrack = off;
      info.wallHit = wall.hit || slot.contact;
      info.speed = slot.car.speed();
      info.wrongWay = wrong;
      info.stuck = scored.info.stuck;
      info.rewardRaw = scored.reward;
      slot.action[0] = steer;
      slot.action[1] = throttle;
      if (scored.done || scored.truncated) {
        this.finished.push({
          lap: slot.lap >= 1,
          lapTime: slot.lap >= 1 ? slot.lapTime : undefined,
          crash: slot.book.crashed || off,
          ret: slot.returning,
          trackId: slot.track.id,
          speed: info.speed,
        });
        if (lapDone) slot.lapTime = 0;
        this.place(slot);
      }
      this.writeObs(i);
      void st;
    }
    return { obs: this.obs, rew: this.rew, done: this.done, trunc: this.trunc, infos: this.infos };
  }

  /** Cheap polyline resolution. Same wall geometry Rapier uses for the driven car. */
  private wallPush(slot: Slot): { hit: boolean; impact: number } {
    resolveWallContact(slot.car, slot.grid, this.wallOut);
    slot.contact = this.wallOut.contact;
    return this.wallOut;
  }

  private writeObs(i: number): void {
    const slot = this.slots[i]!;
    const cap = Math.max(200, Math.round(slot.track.length / 1.5 / CONTROL_DT));
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

  /**
   * Pure-pursuit actions for every car. No argument gives the imitation teacher exactly as before;
   * pass BaselineOpts for the oracle variants, or a controller function to drive with something else.
   */
  baselineActions(ctl?: BaselineOpts | BaselineController): Float32Array {
    const out = new Float32Array(this.n * ACT_DIM);
    for (let i = 0; i < this.n; i++) {
      const slot = this.slots[i];
      if (!slot) continue;
      const cmd =
        typeof ctl === 'function' ? ctl(slot.car, slot.track, slot.hint) : purePursuit(slot.car, slot.track, slot.hint, ctl, slot.mem);
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

  swapTracks(tracks: TrackSpec[], cfgs: EpisodeConfig[]): void {
    for (let i = 0; i < this.n; i++) {
      const slot = this.slots[i];
      if (!slot) continue;
      slot.track = tracks[i % tracks.length]!;
      slot.cfg = { ...cfgs[i % cfgs.length]!, trackId: slot.track.id };
      slot.rays.setTrack(slot.track);
      slot.grid = getWallGrid(slot.track.walls);
    }
  }
}
