/** Fixed observation layout. Keep in sync with sim/OBS_LAYOUT.md. */
export const RAY_COUNT = 21;
export const RAY_MAX = 6;
export const RAY_ARC = (200 * Math.PI) / 180;

export const OBS = {
  RAY0: 0,
  V_LONG: 21,
  V_LAT: 22,
  YAW_RATE: 23,
  SLIP_FRONT: 24,
  SLIP_REAR: 25,
  SLIP_RATIO_REAR: 26,
  PREV_STEER: 27,
  PREV_THROTTLE: 28,
  STEER_ACTUAL: 29,
  LOAD_FRONT: 30,
  LOAD_REAR: 31,
  LAT_OFFSET: 32,
  HEAD_SIN: 33,
  HEAD_COS: 34,
  CURV0: 35,
  TIME_LEFT: 40,
  ROLL: 41,
  PITCH: 42,
  SLIP_RATIO_FRONT: 43,
  SPEED: 44,
  LOAD_FL: 45,
  LOAD_FR: 46,
  PROGRESS: 47,
} as const;

export const V_SCALE = 10;
export const YAW_SCALE = 6;

export const PHYS_HZ = 500;
export const CONTROL_HZ = 50;
export const POLICY_HZ = 25;
export const PHYS_DT = 1 / PHYS_HZ;
export const CONTROL_DT = 1 / CONTROL_HZ;

export const WHEEL_RADIUS = 0.031;
export const G = 9.81;
export const RHO = 1.225;
export const CDA = 0.02;
export const CRR = 0.025;

export const MIN_TRACK_LENGTH = 15;
export const MAX_TRACK_LENGTH = 150;
export const SAMPLE_SPACING = 0.1;
export const MIN_RADIUS = 0.35;
export const BOUNDS = 80;
export const START_STRAIGHT = 1.5;
export const EDGE_GAP = 0.3;
export const WALL_MARGIN = 0.12;

/** 0.6 × typical min turning radius (wheelbase / tan(max steer)). */
export const CAR_TURN_RADIUS = 0.26 / Math.tan(0.48);

export const STAGE_WIDTH = [2.2, 1.5, 1.0, 0.8, 0.75] as const;
export const STAGE_MIN_RADIUS = [6, 3.5, 2.2, 1.1, 0.7] as const;

export const LOOKAHEADS = [0.5, 1.0, 2.0, 3.0, 4.0] as const;
