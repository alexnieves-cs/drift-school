import { expect, test } from '@playwright/test';

// M3 numbers depend on the GPU. Software GL (CI, containers) gives a few fps, so by default the frame-rate and
// long-task numbers are printed, not asserted. Set E2E_STRICT_PERF=1 on a machine with a GPU to assert the SPEC bars.
const STRICT = process.env.E2E_STRICT_PERF === '1';

test('M3: 64 ghost cars frame rate', async ({ page }) => {
  await page.goto('/?view=ghosts&cam=topdown&cars=64');
  await page.waitForSelector('[data-status="racing"]');
  await page.waitForTimeout(1500);
  const fps = await page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        let frames = 0;
        const t0 = performance.now();
        const tick = () => {
          frames++;
          if (performance.now() - t0 >= 3000) resolve((frames * 1000) / (performance.now() - t0));
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }),
  );
  console.log(`[m3] ghost fps with 64 cars: ${fps.toFixed(1)}`);
  expect(fps).toBeGreaterThan(1);
  if (STRICT) expect(fps).toBeGreaterThanOrEqual(50);
});

test('M3: leaderboard rows equal updates, rows replay, and the main thread stays free', async ({ page }) => {
  await page.addInitScript(() => {
    const w = window as unknown as { __long: number[] };
    w.__long = [];
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) w.__long.push(e.duration);
    }).observe({ entryTypes: ['longtask'] });
  });
  await page.goto('/');
  await page.waitForFunction(() => Number(document.querySelector<HTMLElement>('#board')?.dataset.rows ?? 0) >= 3, null, { timeout: 90_000 });
  await page.waitForTimeout(4000);
  const board = await page.evaluate(() => {
    const el = document.querySelector<HTMLElement>('#board')!;
    const rows = [...el.querySelectorAll('tbody tr')].map((tr) => Number((tr as HTMLElement).dataset.gen));
    return { rows: Number(el.dataset.rows), shown: rows.length, newest: Math.max(...rows), oldest: Math.min(...rows) };
  });
  // One row per update: no gaps from dropped messages, none pruned from the table.
  expect(board.shown).toBe(board.rows);
  expect(board.newest - board.oldest + 1).toBe(board.rows);

  await page.locator('#board tbody tr').last().click();
  await expect(page.locator('[data-status]').first()).toContainText('replay gen');

  const long = await page.evaluate(() => (window as unknown as { __long: number[] }).__long);
  const over50 = long.filter((d) => d > 50).length;
  console.log(`[m3] updates seen ${board.rows}; long tasks > 50 ms: ${over50} (max ${Math.max(0, ...long).toFixed(0)} ms)`);
  if (STRICT) expect(over50).toBe(0);
});
