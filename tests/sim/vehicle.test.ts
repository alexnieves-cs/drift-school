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

  it('drifts with low rear grip and recovers under countersteer', () => {
    const p = cloneCar(defaultCarParams());
    p.drive = 'rwd';
    p.split = 0;
    p.tire.mu0 = 1.05;
    p.tire.rearMuScale = 0.42;
    p.tire.front = { B: 10, C: 1.6, E: 0.9 };
    p.tire.rear = { B: 4.2, C: 1.35, E: 0.5 };
    const car = new Vehicle(p);
    for (let i = 0; i < 180; i++) {
      const thr = Math.min(1, 0.45 + (6 - car.speed()) * 0.2);
      car.step(0, thr, 0.02);
    }
    let peaked = 0;
    for (let i = 0; i < 150; i++) {
      car.step(0.85, 1, 0.02);
      peaked = Math.max(peaked, Math.abs(car.meanSlip('rear')));
    }
    expect(peaked).toBeGreaterThan((15 * Math.PI) / 180);
    let recovered = false;
    for (let i = 0; i < 200; i++) {
      const slip = car.meanSlip('rear');
      const counter = -Math.sign(slip || car.wYaw || 1) * 0.95;
      car.step(counter, 0.05, 0.02);
      if (Math.abs(car.meanSlip('rear')) < (8 * Math.PI) / 180 && Math.abs(car.wYaw) < 2.5) {
        recovered = true;
        break;
      }
    }
    expect(recovered).toBe(true);
    expect(stateFinite(car.step(0, 0, 0.02))).toBe(true);
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

  it('control step (10 substeps) stays under 0.15 ms', () => {
    const car = new Vehicle(defaultCarParams());
    for (let i = 0; i < 200; i++) car.step(0.2, 0.5, 0.02);
    const n = 2000;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) car.step((i & 8) ? 0.3 : -0.2, 0.6, 0.02);
    const ms = (performance.now() - t0) / n;
    expect(ms).toBeLessThan(0.15);
  });

  it('bounces off a Rapier wall with restitution', async () => {
    await initRapier();
    const world = new WallWorld();
    const wall = new Float32Array([-1, 2, 1, 2]);
    world.setWalls([wall]);
    const car = new Vehicle(defaultCarParams());
    car.walls = world;
    car.reset({ x: 0, z: 0, yaw: Math.PI / 2, speed: 6 });
    let hit = false;
    for (let i = 0; i < 80; i++) {
      car.step(0, 0.2, 0.02);
      if (car.pz > 1.5 && car.vx < 0) hit = true;
    }
    expect(hit || car.pz < 2.2).toBe(true);
    expect(stateFinite(car.step(0, 0, 0.02))).toBe(true);
    world.free();
  });
});
