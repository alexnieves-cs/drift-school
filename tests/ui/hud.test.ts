import { describe, expect, it } from 'vitest';
import { hudReadout } from '../../src/viz/cameras/hud';
import { Vehicle } from '../../src/sim/car/vehicle';
import { defaultCarParams } from '../../src/sim/car/params';

describe('HUD', () => {
  it('matches car state', () => {
    const car = new Vehicle(defaultCarParams());
    car.step(0.25, 0.8, 0.5);
    const readout = hudReadout({
      speed: car.speed(),
      lap: car.lap,
      lapTime: 10.2,
      bestLap: 9.5,
      steer: 0.25,
      throttle: 0.8,
      slipRear: car.meanSlip('rear'),
    });
    expect(readout.speed).toBe((car.speed() * 3.6).toFixed(1));
    expect(readout.lapTime).toBe('10.20');
    expect(readout.steer).toBeCloseTo(0.25);
    expect(readout.throttle).toBeCloseTo(0.8);
    expect(readout.delta).toBe('+0.70');
  });
});
