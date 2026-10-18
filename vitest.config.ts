import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts', 'tests/**/*.bench.ts'],
    testTimeout: 120_000,
    hookTimeout: 60_000,
    fileParallelism: false,
    poolOptions: {
      threads: { singleThread: true },
    },
  },
});
