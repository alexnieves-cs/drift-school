import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { sCurveLoop } from '../../src/eval/scurve';

test('dev server sends COOP and COEP', async ({ request }) => {
  const res = await request.get('http://127.0.0.1:5173/');
  expect(res.headers()['cross-origin-opener-policy']).toBe('same-origin');
  expect(res.headers()['cross-origin-embedder-policy']).toBe('require-corp');
});

// Software GL (CI, containers) renders far slower than a GPU; set E2E_MIN_FPS to match the machine.
const MIN_FPS = Number(process.env.E2E_MIN_FPS ?? 30);

test('ghost view renders and holds frame rate', async ({ page }, testInfo) => {
  await page.goto('/?view=ghosts&cam=topdown');
  await page.waitForSelector('[data-status="racing"]');
  await page.waitForTimeout(900);
  const fps = Number(await page.evaluate(() => document.body.dataset.fps ?? '0'));
  const shot = await page.screenshot({ path: testInfo.outputPath('race-topdown.png') });
  expect(shot.byteLength).toBeGreaterThan(1000);
  const variance = await page.evaluate(() => {
    const canvas = document.querySelector('#view') as HTMLCanvasElement | null;
    if (!canvas) return 0;
    const probe = document.createElement('canvas');
    probe.width = 64;
    probe.height = 64;
    const ctx = probe.getContext('2d');
    if (!ctx) return 0;
    ctx.drawImage(canvas, 0, 0, 64, 64);
    const pixels = ctx.getImageData(0, 0, 64, 64).data;
    let mean = 0;
    const n = 64 * 64;
    for (let i = 0; i < pixels.length; i += 4) mean += pixels[i]! + pixels[i + 1]! + pixels[i + 2]!;
    mean /= n;
    let v = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      const s = pixels[i]! + pixels[i + 1]! + pixels[i + 2]!;
      v += (s - mean) ** 2;
    }
    return v / n;
  });
  expect(variance).toBeGreaterThan(1);
  expect(fps).toBeGreaterThan(MIN_FPS);
});

test('pov camera view', async ({ page }, testInfo) => {
  await page.goto('/?view=ghosts&cam=pov');
  await page.waitForSelector('[data-status="racing"]');
  await page.waitForTimeout(400);
  await page.screenshot({ path: testInfo.outputPath('pov.png') });
  await expect(page.locator('[data-hud="speed"]')).not.toHaveText('0.0');
});

test('editor accepts a closed stadium stroke', async ({ page }, testInfo) => {
  await page.goto('/?view=editor&quiet=1');
  await page.waitForSelector('#stroke');
  const box = await page.locator('#stroke').boundingBox();
  if (!box) throw new Error('no editor canvas');
  const oval = JSON.parse(readFileSync('tests/fixtures/tracks-good/oval.json', 'utf8')) as { controlPoints: number[] };
  const scale = 0.04;
  const toPx = (x: number, z: number) => ({
    x: box.x + box.width / 2 + x / scale,
    y: box.y + box.height / 2 + z / scale,
  });
  const pts: [number, number][] = [];
  for (let i = 0; i + 1 < oval.controlPoints.length; i += 2) pts.push([oval.controlPoints[i]!, oval.controlPoints[i + 1]!]);
  const start = toPx(pts[0]![0], pts[0]![1]);
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  for (const [x, z] of pts) {
    const p = toPx(x, z);
    await page.mouse.move(p.x, p.y);
  }
  await page.mouse.up();
  await expect(page.locator('.adapt')).toContainText('valid', { timeout: 5000 });
  await page.screenshot({ path: testInfo.outputPath('editor.png') });
});

test('M4: a drawn S-curve is valid and hot-swaps into the learner without touching the weights', async ({ page }, testInfo) => {
  await page.goto('/');
  await page.waitForFunction(() => (document.body.dataset.track ?? '') !== '', null, { timeout: 60_000 });
  const procedural = await page.evaluate(() => document.body.dataset.track);
  expect(procedural).not.toMatch(/^user-/);

  await page.locator('[data-act="editor"]').click();
  await page.waitForSelector('#stroke');
  const box = await page.locator('#stroke').boundingBox();
  if (!box) throw new Error('no editor canvas');
  const scale = 0.04;
  const toPx = (x: number, z: number) => ({ x: box.x + box.width / 2 + x / scale, y: box.y + box.height / 2 + z / scale });
  const loop = sCurveLoop(1);
  const first = toPx(loop[0]![0], loop[0]![1]);
  await page.mouse.move(first.x, first.y);
  await page.mouse.down();
  for (const [x, z] of loop) {
    const p = toPx(x, z);
    await page.mouse.move(p.x, p.y);
  }
  await page.mouse.up();
  await expect(page.locator('.adapt')).toContainText('valid', { timeout: 5000 });
  await page.screenshot({ path: testInfo.outputPath('s-curve-drawn.png') });

  const t0 = Date.now();
  await page.locator('[data-act="train"]').click();
  await page.waitForFunction(() => /^user-/.test(document.body.dataset.track ?? ''), null, { timeout: 15_000 });
  const latencyMs = Date.now() - t0;
  const ack = await page.waitForFunction(() => document.querySelector<HTMLElement>('.adapt')?.dataset.adapt?.startsWith('track user-') ?? false, null, { timeout: 15_000 });
  await ack.dispose();
  const adapt = (await page.locator('.adapt').getAttribute('data-adapt')) ?? '';
  testInfo.annotations.push({ type: 'swap-latency-ms', description: String(latencyMs) });
  expect(latencyMs).toBeLessThan(15_000);
  expect(adapt).toMatch(/weights unchanged/);
  // The worker reports how many envs it moved; every env must be on the new track.
  const envs = Number(/envs (\d+)/.exec(adapt)?.[1] ?? 0);
  expect(envs).toBeGreaterThanOrEqual(8);

  // The URL hash carries the track: a fresh page loads the same drawing and finds it valid.
  const hash = await page.evaluate(() => location.hash);
  expect(hash.length).toBeGreaterThan(8);
  const again = await page.context().newPage();
  await again.goto(`/${hash}`);
  await again.waitForSelector('#stroke', { state: 'attached' });
  await expect(again.locator('.adapt')).toContainText('valid', { timeout: 5000 });
});
