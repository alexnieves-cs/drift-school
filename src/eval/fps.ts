import type { Section } from './types';

/** Frame rate and main-thread budgets need a real browser, so this section only points at where they are measured. */
export const fps: Section = async () => [
  {
    milestone: 'M3',
    test: '64 ghost cars frame rate',
    target: '>= 50 fps',
    measured: 'not measured here',
    pass: null,
    note: 'Needs a browser with a GPU. `E2E_STRICT_PERF=1 npm run test:e2e -- m3` asserts it (and zero long tasks over 50 ms); without the flag the run prints the measured numbers as `[m3]` lines. Software GL gives about 7 fps.',
  },
];
