import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    // Wall-clock budgets live in tests/perf and run with `npm run bench`.
    include: ['tests/**/*.test.ts'],
    testTimeout: 120_000,
    hookTimeout: 60_000,
    fileParallelism: false,
    poolOptions: {
      threads: { singleThread: true },
    },
  },
});
