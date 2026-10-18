// Pure-pursuit baseline laps the real vehicle model on a hairpin track, one lap per grip level.
// This is a recording stage: the learned policy is not driving here.
import { ThreeRenderer } from '../viz/renderer';
import { Hud } from '../viz/cameras/hud';
import { buildTrackFromPoints } from '../track/build';
import { purePursuit } from '../track/baseline';
import { nearestSample } from '../track/sample';
import { Vehicle } from '../sim/car/vehicle';
import { defaultCarParams, cloneCar } from '../sim/car/params';
import { CONTROL_DT } from '../shared/constants';
import type { CameraMode, TrackSpec } from '../shared/types';

const params = new URLSearchParams(location.search);
const mus = (params.get('mus') ?? '1.1,0.4').split(',').map(Number);
const manual = params.get('manual') === '1';
const cam = (params.get('cam') ?? 'pov') as CameraMode;

/** A valid closed loop with a tight U-turn at one end (checked by validateTrack in tests). */
export const HAIRPIN = [
  [0, -5], [4, -5], [8, -5], [11.5, -4.4], [13.8, -2.6], [14.6, 0], [13.8, 2.6], [11.5, 4.4],
  [8, 5], [4, 5], [0, 5], [-4, 5], [-8, 5], [-12, 3.6], [-14, 0], [-12, -3.6], [-8, -5], [-4, -5], [0, -5],
].flat();

const root = document.querySelector('#showcase') as HTMLElement;
root.innerHTML = `<canvas id="view"></canvas><div id="hud-host"></div><div id="caption" class="caption"></div>`;
const canvas = document.querySelector('#view') as HTMLCanvasElement;
const caption = document.querySelector('#caption') as HTMLElement;
const renderer = new ThreeRenderer(canvas);
const hud = new Hud(document.querySelector('#hud-host') as HTMLElement);

await renderer.init(params.get('quality') === 'high' ? 'high' : 'low');
renderer.setCamera(cam);

interface LapResult {
  mu: number;
  finished: boolean;
  lapTime: number;
  maxOffset: number;
  topSpeed: number;
}
const results: LapResult[] = [];

let stage = -1;
let track: TrackSpec;
let car: Vehicle;
let hint = 0;
let prevS = 0;
let time = 0;
let maxOff = 0;
let top = 0;
let done = false;
let hold = 0;

function startLap(i: number): void {
  stage = i;
  const mu = mus[i]!;
  track = buildTrackFromPoints(Float32Array.from(HAIRPIN), { halfWidth: 1, mu, source: 'user', id: `hairpin-mu${mu}` });
  const p = cloneCar(defaultCarParams());
  p.tire.mu0 = (1.15 * mu) / 1.1;
  car = new Vehicle(p);
  const s0 = track.samples[track.startIndex]!;
  car.reset({ x: s0.x, z: s0.z, yaw: Math.atan2(s0.tz, s0.tx), speed: 1.2 });
  hint = track.startIndex;
  prevS = s0.s;
  time = 0;
  maxOff = 0;
  top = 0;
  renderer.loadTrack(track);
  renderer.resize();
  hold = 0;
  const label = mu >= 1 ? 'dry' : mu >= 0.6 ? 'wet' : 'ice';
  caption.textContent = `pure-pursuit baseline · ${label} grip, mu ${mu.toFixed(2)}`;
}

function control(): void {
  if (done) return;
  if (hold > 0) {
    hold--;
    if (hold === 0) {
      if (stage + 1 < mus.length) startLap(stage + 1);
      else done = true;
    }
    return;
  }
  const cmd = purePursuit(car, track, hint);
  hint = cmd.hint;
  car.step(cmd.steer, cmd.throttle, CONTROL_DT);
  time += CONTROL_DT;
  const q = nearestSample(track, car.px, car.pz, hint);
  hint = q.index;
  maxOff = Math.max(maxOff, Math.abs(q.lateral));
  top = Math.max(top, car.speed());
  const lapped = q.sample.s < prevS - track.length * 0.5 && prevS > track.length * 0.4;
  prevS = q.sample.s;
  if (lapped || time > 60) {
    results.push({ mu: mus[stage]!, finished: lapped, lapTime: time, maxOffset: maxOff, topSpeed: top });
    caption.textContent = `${caption.textContent} · lap ${time.toFixed(1)} s`;
    hold = 40;
  }
}

function draw(): void {
  const rear = car.meanSlip('rear');
  const hero = {
    x: car.px, y: car.py, z: car.pz, yaw: car.yaw, pitch: car.pitch, roll: car.roll,
    speed: car.speed(), steer: car.steer / 0.48, throttle: 0.6, slipRear: rear,
    lap: 0, lapTime: time, bestLap: null,
  };
  renderer.frame(0, -1, hero);
  hud.update(
    { speed: hero.speed, lap: 0, lapTime: time, bestLap: null, steer: hero.steer, throttle: hero.throttle, slipRear: rear },
    track.samples.filter((_, i) => i % 6 === 0),
    hero,
  );
}

startLap(0);
draw();

declare global {
  interface Window {
    __showcase: { advance: (steps: number) => boolean; results: LapResult[] };
  }
}

/** Steps the sim by `steps` control ticks (0.02 s each), renders once, and returns true when finished. */
window.__showcase = {
  advance(steps: number): boolean {
    for (let i = 0; i < steps; i++) control();
    draw();
    return done;
  },
  results,
};

if (!manual) {
  const loop = () => {
    control();
    draw();
    if (!done) requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
