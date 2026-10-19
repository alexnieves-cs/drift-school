import { expect, test, type Page } from '@playwright/test';
import { sCurveStroke } from '../../src/editor/pipeline';

/**
 * M4 in a real browser: draw an S-curve with the mouse on the editor canvas, press "train on this",
 * and read what the learner worker reports back. Nothing here reaches into the app's modules;
 * everything is read from the DOM the page exposes (data attributes on <body> and the panel).
 */

/** World meters per canvas pixel, as main.ts mounts the editor. */
const SCALE = 0.04;
const SWAP_BUDGET_MS = 15_000;

async function drawSCurve(page: Page): Promise<void> {
  await page.waitForSelector('#stroke');
  const box = await page.locator('#stroke').boundingBox();
  if (!box) throw new Error('no editor canvas');
  const toPx = (x: number, z: number) => ({
    x: box.x + box.width / 2 + x / SCALE,
    y: box.y + box.height / 2 + z / SCALE,
  });
  // A mouse event every ~10 px (0.4 m at this scale), ending a few px short of the start.
  const stroke = sCurveStroke().filter((_, i) => i % 2 === 0);
  const first = toPx(stroke[0]!.x, stroke[0]!.z);
  await page.mouse.move(first.x, first.y);
  await page.mouse.down();
  for (const p of stroke) {
    const px = toPx(p.x, p.z);
    await page.mouse.move(px.x, px.y);
  }
  await page.mouse.up();
}

const body = (page: Page, key: string) => page.evaluate((k) => document.body.dataset[k] ?? '', key);

test('draw an S-curve, hot-swap it, all envs on the new track within 15 s', async ({ page }) => {
  await page.goto('/?view=editor');
  await page.waitForSelector('[data-status="racing"]', { timeout: 90_000 });
  const before = await body(page, 'track');
  expect(before).toMatch(/^gen-/);

  await drawSCurve(page);
  await expect(page.locator('.adapt')).toContainText('valid', { timeout: 5000 });

  const clicked = Date.now();
  await page.locator('[data-act="train"]').click();
  // UI feedback inside 100 ms: the status line and the adaptation panel change on the click itself.
  await expect(page.locator('[data-status="adapt"]')).toBeVisible({ timeout: 100 });
  await expect(page.locator('section.adaptation')).toBeVisible({ timeout: 100 });

  const userTrack = await body(page, 'userTrack');
  expect(userTrack).toMatch(/^user-[0-9a-f]{8}$/);

  // All envs report the new track id: read from the learner's ghost stream, not from an ack.
  await page.waitForFunction(() => !!document.body.dataset.allOnUserMs, null, { timeout: SWAP_BUDGET_MS, polling: 50 });
  const wall = Date.now() - clicked;
  const allOnUserMs = Number(await body(page, 'allOnUserMs'));
  expect(allOnUserMs).toBeLessThanOrEqual(SWAP_BUDGET_MS);
  expect(wall).toBeLessThanOrEqual(SWAP_BUDGET_MS + 1000);
  const total = Number(await body(page, 'envTotal'));
  expect(total).toBeGreaterThanOrEqual(8);

  // The worker's own acknowledgement agrees, and the weights are byte-identical at the swap.
  await expect.poll(() => body(page, 'swapAckTrack'), { timeout: 5000 }).toBe(userTrack);
  expect(await body(page, 'swapAckEnvs')).toBe(`${total}/${total}`);
  expect(await body(page, 'swapHashEqual')).toBe('true');
  expect(Number(await body(page, 'swapAckMs'))).toBeLessThanOrEqual(SWAP_BUDGET_MS);

  // The URL hash carries the track, and the adaptation panel moves on to the zero-shot result.
  expect(new URL(page.url()).hash.length).toBeGreaterThan(20);
  await expect(page.locator('section.adaptation')).toHaveAttribute('data-zero-shot', /^\d\.\d+$/, { timeout: 60_000 });
  await expect(page.locator('section.adaptation')).toHaveAttribute('data-phase', 'tune', { timeout: 60_000 });
  await page.screenshot({ path: 'docs/media/hotswap.png' });
});

test('the URL hash round-trips the drawn track to the same track id', async ({ page, browser }) => {
  await page.goto('/?view=editor&quiet=1');
  await drawSCurve(page);
  await expect(page.locator('.adapt')).toContainText('valid', { timeout: 5000 });
  await page.locator('[data-act="train"]').click();
  const id = await body(page, 'userTrack');
  expect(id).toMatch(/^user-[0-9a-f]{8}$/);
  const url = page.url();
  expect(new URL(url).hash.length).toBeGreaterThan(20);

  const other = await browser.newPage();
  await other.goto(url);
  await other.waitForSelector('#stroke');
  await expect(other.locator('.adapt')).toContainText('valid', { timeout: 5000 });
  await other.locator('[data-act="train"]').click();
  expect(await other.evaluate(() => document.body.dataset.userTrack)).toBe(id);
  await other.close();
});

test('@perf completion on the drawn track reaches 80% within 2 minutes of fine-tuning', async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto('/?view=editor');
  await page.waitForSelector('[data-status="racing"]', { timeout: 90_000 });
  await drawSCurve(page);
  await expect(page.locator('.adapt')).toContainText('valid', { timeout: 5000 });
  await page.locator('[data-act="train"]').click();
  await page.waitForFunction(() => document.querySelector('section.adaptation')?.getAttribute('data-phase') === 'tune', null, {
    timeout: 60_000,
  });
  // The panel's completion is the policy's own sampled-action episodes on the drawn track (the
  // pure-pursuit teacher is excluded), over the last 16, so it is stricter than the deterministic
  // `npm run eval -- --only=hotswap` number.
  await page.waitForFunction(
    () => Number(document.querySelector('section.adaptation')?.getAttribute('data-completion') ?? '0') >= 0.8,
    null,
    { timeout: 120_000, polling: 500 },
  );
  const completion = Number(await page.locator('section.adaptation').getAttribute('data-completion'));
  expect(completion).toBeGreaterThanOrEqual(0.8);
});
