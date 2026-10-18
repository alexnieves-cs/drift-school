import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';

test('dev server sends COOP and COEP', async ({ request }) => {
  const res = await request.get('/');
  expect(res.headers()['cross-origin-opener-policy']).toBe('same-origin');
  expect(res.headers()['cross-origin-embedder-policy']).toBe('require-corp');
});

test('ghost view renders and holds frame rate', async ({ page }) => {
  await page.goto('/?view=ghosts&cam=topdown');
  await page.waitForSelector('[data-status="racing"]');
  await page.waitForTimeout(900);
  const fps = Number(await page.evaluate(() => document.body.dataset.fps ?? '0'));
  const shot = await page.screenshot({ path: 'docs/media/race-topdown.png' });
  expect(shot.byteLength).toBeGreaterThan(1000);
  const variance = await page.evaluate(() => {
    // The drawing buffer is not preserved between tasks, so render and read in the same task.
    (window as unknown as { __drift: { viz: { render: () => void } } }).__drift.viz.render();
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
  expect(fps).toBeGreaterThan(30);
});

test('pov camera view', async ({ page }) => {
  await page.goto('/?view=ghosts&cam=pov');
  await page.waitForSelector('[data-status="racing"]');
  await page.waitForTimeout(400);
  await page.screenshot({ path: 'docs/media/pov.png' });
  await expect(page.locator('[data-hud="speed"]')).not.toHaveText('0.0');
});

test('editor accepts a closed stadium stroke', async ({ page }) => {
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
  await page.screenshot({ path: 'docs/media/editor.png' });
});

interface Drift {
  viz: { cameraMode: string; preset: string };
  frameStats: () => { frames: number; total: { max: number; mean: number }; calls: number; triangles: number };
  resetFrameStats: () => void;
  metricsCount: () => number;
  boardRows: () => number;
  probeReplay: (gen: number) => number[] | null;
}
type DriftWindow = { __drift: Drift };

test('64 ghosts draw in few calls and the JS frame stays cheap', async ({ page }) => {
  await page.goto('/?view=ghosts&cars=64&cam=topdown');
  await page.waitForSelector('[data-status="racing"]');
  await page.waitForTimeout(1500);
  await page.evaluate(() => (window as unknown as DriftWindow).__drift.resetFrameStats());
  await page.waitForTimeout(2500);
  const stats = await page.evaluate(() => (window as unknown as DriftWindow).__drift.frameStats());
  expect(stats.frames).toBeGreaterThan(5);
  // 64 cars are one instanced draw; the whole frame, PIP included, stays in a handful of calls.
  expect(stats.calls).toBeLessThanOrEqual(50);
  expect(stats.triangles).toBeLessThan(100_000);
  // JS side of the frame, independent of the GPU. A single task over 50 ms would be a long task.
  expect(stats.total.max).toBeLessThan(50);
  expect(stats.total.mean).toBeLessThan(8);
});

test('camera keys and quality buttons work', async ({ page }) => {
  await page.goto('/?view=ghosts&cars=16');
  await page.waitForSelector('[data-status="racing"]');
  const mode = () => page.evaluate(() => (window as unknown as DriftWindow).__drift.viz.cameraMode);
  const preset = () => page.evaluate(() => (window as unknown as DriftWindow).__drift.viz.preset);
  await page.keyboard.press('2');
  expect(await mode()).toBe('chase');
  await page.keyboard.press('3');
  expect(await mode()).toBe('pov');
  await page.keyboard.press('1');
  expect(await mode()).toBe('topdown');
  await page.click('[data-q="high"]');
  await expect.poll(preset).toBe('high');
  await page.waitForTimeout(800);
  const fps = Number(await page.evaluate(() => document.body.dataset.fps ?? '0'));
  expect(fps).toBeGreaterThan(0);
  await page.click('[data-q="low"]');
  await expect.poll(preset).toBe('low');
});

test('leaderboard has one row per PPO update and a row replays the same ghost lap twice', async ({ page }) => {
  await page.goto('/?cars=8&quality=low');
  await page.waitForFunction(() => document.querySelectorAll('table.board tbody tr').length >= 2, null, { timeout: 150_000 });
  const snap = await page.evaluate(() => {
    const d = (window as unknown as DriftWindow).__drift;
    return { updates: d.metricsCount(), rows: d.boardRows(), dom: document.querySelectorAll('table.board tbody tr').length };
  });
  expect(snap.rows).toBe(snap.updates);
  expect(snap.dom).toBe(snap.updates);
  const gen = snap.updates;
  for (let k = 0; k < 2; k++) {
    await page.click(`table.board tbody tr[data-gen="${gen}"]`);
    await page.waitForFunction((g) => document.body.dataset.replay === String(g), gen, { timeout: 5000 });
    await page.waitForFunction(() => document.body.dataset.replay === '', null, { timeout: 15_000 });
  }
  const probes = await page.evaluate((g) => {
    const d = (window as unknown as DriftWindow).__drift;
    return { a: d.probeReplay(g), b: d.probeReplay(g) };
  }, gen);
  expect(probes.a?.length ?? 0).toBeGreaterThan(8);
  expect(probes.b).toEqual(probes.a);
});
