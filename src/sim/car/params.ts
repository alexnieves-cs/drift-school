import type { CarParams } from '../../shared/types';

export function defaultCarParams(): CarParams {
  const mass = 1.6;
  const L = 0.32;
  const W = 0.18;
  const H = 0.05;
  return {
    mass,
    cgHeight: 0.045,
    wheelbase: 0.26,
    track: 0.2,
    yawInertia: (mass / 12) * (L * L + W * W),
    pitchInertia: (mass / 12) * (H * H + L * L),
    rollInertia: (mass / 12) * (H * H + W * W),
    tire: {
      front: { B: 8, C: 1.5, E: 0.92 },
      rear: { B: 8, C: 1.5, E: 0.92 },
      mu0: 1.15,
      rearMuScale: 1,
      loadSens: 0.1,
    },
    susp: { k: 700, c: 55, travel: 0.02, arb: 120 },
    motor: { torque: 0.32, maxRpm: 4500, tau: 0.03 },
    steer: { maxAngle: 0.48, rate: 10.472, offset: 0 },
    drive: '4wd',
    split: 0.4,
  };
}

export function cloneCar(p: CarParams): CarParams {
  return {
    ...p,
    tire: { ...p.tire, front: { ...p.tire.front }, rear: { ...p.tire.rear } },
    susp: { ...p.susp },
    motor: { ...p.motor },
    steer: { ...p.steer },
  };
}
