import { describe, expect, it } from 'vitest';
import { ChaseCamera } from '../../src/viz/cameras/chase';
import { generateTrack } from '../../src/track/generate';
import { purePursuit, simulateBaseline } from '../../src/track/baseline';
import { Vehicle } from '../../src/sim/car/vehicle';
import { defaultCarParams } from '../../src/sim/car/params';
import { CONTROL_DT } from '../../src/shared/constants';

describe('chase camera', () => {
  it('stays outside walls for a scripted minute', () => {
    const track = generateTrack({ seed: 2, stage: 0 });
    const lap = simulateBaseline(track, 30);
    expect(lap.finished).toBe(true);
    const cam = new ChaseCamera();
    const car = new Vehicle(defaultCarParams());
    const s0 = track.samples[track.startIndex] ?? track.samples[0]!;
    car.reset({ x: s0.x, z: s0.z, yaw: Math.atan2(s0.tz, s0.tx), speed: 1.2 });
    let hint = track.startIndex;
    let inside = 0;
    const steps = Math.round(60 / CONTROL_DT);
    for (let i = 0; i < steps; i++) {
      const cmd = purePursuit(car, track, hint);
      hint = cmd.hint;
      car.step(cmd.steer, cmd.throttle, CONTROL_DT);
      cam.update({ x: car.px, y: car.py, z: car.pz, yaw: car.yaw, speed: car.speed() }, track, CONTROL_DT);
      if (cam.insideWall(track)) inside++;
    }
    expect(inside).toBe(0);
  });
});
