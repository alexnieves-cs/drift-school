import { describe, expect, it } from 'vitest';
import { makeRng } from '../../src/shared/rng';
import { Vehicle, stateFinite } from '../../src/sim/car/vehicle';
import { cloneCar, defaultCarParams } from '../../src/sim/car/params';
import { analyticTopSpeed } from '../../src/sim/car/drivetrain';
import { initRapier, WallWorld } from '../../src/sim/world';

function run(car: Vehicle, steer: number, throttle: number, seconds: number, dt = 0.02): void {
  const n = Math.round(seconds / dt);
  for (let i = 0; i < n; i++) car.step(steer, throttle, dt);
}

describe('vehicle', () => {
  it('launch reaches analytic top speed within 5%', () => {
    const params = defaultCarParams();
    const vRef = analyticTopSpeed(params);
    const car = new Vehicle(params);
    run(car, 0, 1, 25);
    const v = car.speed();
    expect(vRef).toBeGreaterThan(4);
    expect(Math.abs(v - vRef) / vRef).toBeLessThan(0.05);
  });

  it('braking from 5 m/s shifts at least 10% of load onto the front axle', () => {
    const car = new Vehicle(defaultCarParams());
    run(car, 0, 0, 1.0);
    const rest = car.frontLoadShare();
    run(car, 0, 1, 4);
    expect(car.speed()).toBeGreaterThan(5);
    let peak = rest;
    for (let i = 0; i < 80; i++) {
      car.step(0, -1, 0.02);
      if (car.speed() < 5.2 && car.speed() > 1.5) peak = Math.max(peak, car.frontLoadShare());
    }
    expect(peak - rest).toBeGreaterThanOrEqual(0.1);
  });

  it('steady circle understeer sign follows the axle setup', () => {
    const under = cloneCar(defaultCarParams());
    under.tire.front = { B: 4.5, C: 1.4, E: 0.9 };
    under.tire.rear = { B: 12, C: 1.6, E: 0.95 };
    under.tire.mu0 = 1.3;
    const over = cloneCar(defaultCarParams());
    over.tire.front = { B: 12, C: 1.6, E: 0.95 };
    over.tire.rear = { B: 3.2, C: 1.3, E: 0.6 };
    over.tire.rearMuScale = 0.45;
    over.tire.mu0 = 1.2;
    over.drive = 'rwd';
    over.split = 0;

    const measure = (p: typeof under) => {
      const car = new Vehicle(p);
      let accF = 0;
      let accR = 0;
      let n = 0;
      for (let i = 0; i < 400; i++) {
        const thr = Math.max(0.15, Math.min(0.85, 0.4 + (3.2 - car.speed()) * 0.35));
        car.step(0.55, thr, 0.02);
        if (i > 250 && car.speed() > 1.5) {
          accF += Math.abs(car.meanSlip('front'));
          accR += Math.abs(car.meanSlip('rear'));
          n++;
        }
      }
      return (accF - accR) / Math.max(1, n);
    };
    expect(measure(under)).toBeGreaterThan(0);
    expect(measure(over)).toBeLessThan(0);
  });

  it('step-steer yaw-rate overshoot stays bounded', () => {
    const car = new Vehicle(defaultCarParams());
    for (let i = 0; i < 200; i++) {
      const thr = Math.max(0, Math.min(1, 0.35 + (4 - car.speed()) * 0.4));
      car.step(0, thr, 0.02);
    }
    const rates: number[] = [];
    for (let i = 0; i < 200; i++) {
      const thr = Math.max(0, Math.min(1, 0.3 + (4 - car.speed()) * 0.3));
      car.step(0.45, thr, 0.02);
      rates.push(car.wYaw);
    }
    const steady = rates.slice(-40).reduce((a, b) => a + b, 0) / 40;
    const peak = Math.max(...rates.map((r) => Math.abs(r)));
    expect(Math.abs(steady)).toBeGreaterThan(0.15);
    expect(peak / Math.abs(steady)).toBeLessThan(1.85);
  });

  it('holds a sustained drift with low rear grip and recovers when steer is released', () => {
    const settle = (rearMu: number) => {
      const p = cloneCar(defaultCarParams());
      p.tire.rearMuScale = rearMu;
      const car = new Vehicle(p);
      for (let i = 0; i < 250; i++) car.step(0, Math.min(1, 0.5 + (5 - car.speed()) * 0.3), 0.02);
      return car;
    };
    const rad = (d: number) => (d * Math.PI) / 180;

    // Reduced rear grip: the rear axle stays sideways at a steady angle, the car keeps moving.
    const car = settle(0.8);
    const slips: number[] = [];
    let peakYawRate = 0;
    for (let i = 0; i < 150; i++) {
      car.step(0.4, 0.7, 0.02);
      slips.push(Math.abs(car.meanSlip('rear')));
      peakYawRate = Math.max(peakYawRate, Math.abs(car.wYaw));
    }
    const late = slips.slice(-50);
    expect(Math.min(...late)).toBeGreaterThan(rad(25));
    expect(Math.max(...late)).toBeLessThan(rad(65));
    expect(Math.max(...late) - Math.min(...late)).toBeLessThan(rad(10));
    expect(peakYawRate).toBeLessThan(4);
    expect(car.speed()).toBeGreaterThan(1.5);

    // Releasing the steering brings the rear back under control.
    let recovered = false;
    for (let i = 0; i < 200 && !recovered; i++) {
      car.step(0, 0.3, 0.02);
      recovered = Math.abs(car.meanSlip('rear')) < rad(8) && Math.abs(car.wYaw) < 1.5;
    }
    expect(recovered).toBe(true);
    expect(stateFinite(car.step(0, 0, 0.02))).toBe(true);

    // Negative control: far less rear grip is a spin-out, not a drift (yaw sweeps more than two turns).
    const spin = settle(0.5);
    const yaw0 = spin.yaw;
    for (let i = 0; i < 150; i++) spin.step(0.7, 0.7, 0.02);
    expect(Math.abs(spin.yaw - yaw0)).toBeGreaterThan(4 * Math.PI);
  });

  it('matches a twin given the same controls', () => {
    const rng = makeRng(99);
    const a = new Vehicle(defaultCarParams());
    const b = new Vehicle(defaultCarParams());
    for (let i = 0; i < 300; i++) {
      const s = rng.range(-1, 1);
      const t = rng.range(-1, 1);
      const sa = a.step(s, t, 0.02);
      const sb = b.step(s, t, 0.02);
      expect(sa.pos[0]).toBe(sb.pos[0]);
      expect(sa.pos[2]).toBe(sb.pos[2]);
      expect(sa.yawRate).toBe(sb.yawRate);
    }
  });

  it('survives 1e6 random substeps without NaN', () => {
    const rng = makeRng(123456);
    const car = new Vehicle(defaultCarParams());
    const dt = 1 / 500;
    for (let i = 0; i < 1_000_000; i++) {
      car.step(rng.range(-1, 1), rng.range(-1, 1), dt);
      if ((i & 4095) === 0) {
        const s = car.step(0, 0, dt);
        expect(stateFinite(s)).toBe(true);
      }
    }
    expect(stateFinite(car.step(0, 0, dt))).toBe(true);
  });

  it('bounces off a Rapier wall with restitution', async () => {
    await initRapier();
    const world = new WallWorld();
    const wall = new Float32Array([-1, 2, 1, 2]);
    world.setWalls([wall]);
    const car = new Vehicle(defaultCarParams());
    car.walls = world;
    car.reset({ x: 0, z: 0, yaw: Math.PI / 2, speed: 6 });
    let rebounded = false;
    let maxZ = -Infinity;
    for (let i = 0; i < 80; i++) {
      car.step(0, 0.2, 0.02);
      maxZ = Math.max(maxZ, car.pz);
      if (car.vx < -0.5) rebounded = true;
    }
    expect(rebounded).toBe(true);
    // The wall sits at z = 2; the chassis may graze it but must not tunnel through.
    expect(maxZ).toBeLessThan(2.1);
    expect(stateFinite(car.step(0, 0, 0.02))).toBe(true);
    world.free();
  });
});
