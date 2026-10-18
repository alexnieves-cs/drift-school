import type { Section } from './types';

export const fps: Section = async () => [
  { milestone: 'M3', test: '64 ghost cars frame rate', target: '>= 50 fps', measured: 'not measured', pass: null },
];
