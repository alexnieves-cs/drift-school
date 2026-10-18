import tseslint from 'typescript-eslint';

const domBanned = [
  'window',
  'document',
  'localStorage',
  'sessionStorage',
  'indexedDB',
  'navigator',
  'HTMLElement',
  'requestAnimationFrame',
];

export default tseslint.config(
  {
    ignores: ['dist/**', 'node_modules/**', 'docs/**', 'test-results/**', 'playwright-report/**', 'scripts/**'],
  },
  ...tseslint.configs.recommended,
  {
    files: ['src/track/**/*.ts', 'src/sim/**/*.ts', 'src/rl/**/*.ts', 'src/workers/env*.ts', 'src/workers/learner*.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          paths: [
            { name: 'three', message: 'Renderer imports are banned outside viz/ and ui/.' },
            { name: 'uplot', message: 'Chart imports are banned outside ui/.' },
          ],
          patterns: [
            { group: ['**/ui/**', '**/viz/**'], message: 'Sim, track, rl, and env workers stay DOM-free.' },
          ],
        },
      ],
      'no-restricted-globals': ['error', ...domBanned],
      'no-restricted-properties': ['error', { object: 'Math', property: 'random', message: 'Use shared/rng.ts.' }],
    },
  },
  {
    files: ['src/rl/checkpoint.ts'],
    rules: {
      'no-restricted-globals': 'off',
    },
  },
);
