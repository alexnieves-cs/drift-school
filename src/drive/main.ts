import { ThreeRenderer } from '../viz/renderer';
import { Hud } from '../viz/cameras/hud';
import { generateTrack } from '../track/generate';
import { Vehicle } from '../sim/car/vehicle';
import { defaultCarParams } from '../sim/car/params';
import { WallWorld, initRapier } from '../sim/world';
import { CONTROL_DT } from '../shared/constants';

const root = document.querySelector('#drive') as HTMLElement;
root.innerHTML = `<canvas id="view"></canvas><pre id="drive-log"></pre><div id="hud-host"></div>`;
const canvas = document.querySelector('#view') as HTMLCanvasElement;
const log = document.querySelector('#drive-log') as HTMLElement;
const renderer = new ThreeRenderer(canvas);
const hud = new Hud(document.querySelector('#hud-host') as HTMLElement);
const track = generateTrack({ seed: 3, stage: 0 });
const car = new Vehicle(defaultCarParams());
const keys = new Set<string>();

window.addEventListener('keydown', (e) => keys.add(e.key.toLowerCase()));
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

await initRapier();
const walls = new WallWorld();
walls.setWalls(track.walls);
car.walls = walls;
const s0 = track.samples[track.startIndex] ?? track.samples[0]!;
car.reset({ x: s0.x, z: s0.z, yaw: Math.atan2(s0.tz, s0.tx), speed: 0 });

await renderer.init('low');
renderer.loadTrack(track);
renderer.setCamera('chase');
renderer.resize();
window.addEventListener('resize', () => renderer.resize());

let steer = 0;
let throttle = 0;
function frame(): void {
  const left = keys.has('a') || keys.has('arrowleft');
  const right = keys.has('d') || keys.has('arrowright');
  const go = keys.has('w') || keys.has('arrowup');
  const stop = keys.has('s') || keys.has('arrowdown');
  steer = left ? -1 : right ? 1 : steer * 0.85;
  throttle = go ? 1 : stop ? -1 : 0;
  if (keys.has('1')) renderer.setCamera('topdown');
  if (keys.has('2')) renderer.setCamera('chase');
  if (keys.has('3')) renderer.setCamera('pov');
  const state = car.step(steer, throttle, CONTROL_DT);
  const speed = car.speed();
  log.textContent = `x ${state.pos[0].toFixed(2)} z ${state.pos[2].toFixed(2)} v ${speed.toFixed(2)} steer ${state.steer.toFixed(2)} Fz ${Array.from(state.wheelLoad).map((n) => n.toFixed(2)).join(' ')}`;
  renderer.frame(0, -1, {
    x: car.px,
    y: car.py,
    z: car.pz,
    yaw: car.yaw,
    pitch: car.pitch,
    roll: car.roll,
    speed,
    steer,
    throttle,
    slipRear: car.meanSlip('rear'),
    lap: car.lap,
    lapTime: car.lapTime,
  });
  hud.update({
    speed,
    lap: car.lap,
    lapTime: car.lapTime,
    bestLap: null,
    steer,
    throttle,
    slipRear: car.meanSlip('rear'),
  });
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
