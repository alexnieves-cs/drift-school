import { defineConfig } from '@playwright/test';
import { CHROMIUM_GL_ARGS, findChromium } from './src/eval/chromium';

// E2E_PORT lets parallel checkouts run their own dev server instead of reusing someone else's.
const port = Number(process.env.E2E_PORT ?? 5173);
const executablePath = findChromium() ?? undefined;

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    headless: true,
    // The headless shell has no locale of its own and reports "en-US@posix", which Intl rejects.
    locale: 'en-US',
    viewport: { width: 1280, height: 720 },
    launchOptions: {
      executablePath,
      args: CHROMIUM_GL_ARGS,
    },
  },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
