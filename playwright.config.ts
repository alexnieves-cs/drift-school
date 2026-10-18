import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: 'http://127.0.0.1:5173',
    headless: true,
    viewport: { width: 1280, height: 720 },
    launchOptions: {
      // Set PW_CHROMIUM_PATH to reuse a system Chromium instead of the pinned Playwright download.
      executablePath: process.env.PW_CHROMIUM_PATH || undefined,
      args: ['--enable-webgl', '--use-gl=angle', '--ignore-gpu-blocklist'],
    },
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 5173',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
