/**
 * `npm run media` retakes the screenshots and GIF in docs/media with Playwright.
 * Set PW_CHROMIUM_PATH to reuse a system Chromium. Needs a running dev server (`npm run dev`) on :5173.
 * Software GL renders a few frames per second, so the GIF is choppy there; run it on a GPU machine for a smooth one.
 */
import { mkdirSync, rmSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { chromium } from '@playwright/test';
import { sCurveLoop } from '../src/eval/scurve';

const BASE = process.env.BASE_URL ?? 'http://127.0.0.1:5173';
const OUT = 'docs/media';
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({
  executablePath: process.env.PW_CHROMIUM_PATH || undefined,
  args: ['--enable-webgl', '--use-gl=angle', '--ignore-gpu-blocklist'],
});
const open = async (query) => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await page.goto(`${BASE}/${query}`);
  return page;
};

// Training view: wait until the leaderboard and charts have real rows.
{
  const page = await open('?cam=topdown');
  await page.waitForFunction(() => Number(document.querySelector('#board')?.dataset.rows ?? 0) >= 12, null, { timeout: 240_000 });
  await page.screenshot({ path: `${OUT}/train.png` });

  // Draw the S-curve on top of the training page and hot-swap onto it.
  await page.locator('[data-act="editor"]').click();
  const box = await page.locator('#stroke').boundingBox();
  const px = (x, z) => ({ x: box.x + box.width / 2 + x / 0.04, y: box.y + box.height / 2 + z / 0.04 });
  const loop = sCurveLoop(1);
  const first = px(...loop[0]);
  await page.mouse.move(first.x, first.y);
  await page.mouse.down();
  for (const [x, z] of loop) {
    const p = px(x, z);
    await page.mouse.move(p.x, p.y);
  }
  await page.mouse.up();
  await page.waitForFunction(() => document.querySelector('.adapt')?.textContent?.includes('valid'));
  await page.screenshot({ path: `${OUT}/editor.png` });
  await page.locator('[data-act="train"]').click();
  await page.waitForFunction(() => document.querySelector('.adapt')?.dataset.adapt?.startsWith('track user-'));
  // Let the adaptation card fill in.
  await page.waitForFunction(() => /drawn track: first/.test(document.querySelector('.adapt')?.textContent ?? ''), null, { timeout: 180_000 });
  await page.screenshot({ path: `${OUT}/adapt.png` });
  await page.close();
}

for (const [cam, file] of [['pov', 'pov.png'], ['chase', 'chase.png'], ['topdown', 'race-topdown.png']]) {
  const page = await open(`?view=ghosts&cam=${cam}`);
  await page.waitForSelector('[data-status="racing"]');
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${OUT}/${file}` });
  await page.close();
}

// GIF: a few seconds of the top-down ghost view.
{
  const page = await open('?view=ghosts&cam=topdown&quiet=1');
  await page.waitForSelector('[data-status="racing"]');
  const tmp = '.media-frames';
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(tmp);
  for (let i = 0; i < 40; i++) {
    await page.screenshot({ path: `${tmp}/f${String(i).padStart(3, '0')}.png` });
    await page.waitForTimeout(80);
  }
  await page.close();
  if (readdirSync(tmp).length) {
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', '8', '-i', `${tmp}/f%03d.png`, '-vf', 'scale=800:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse', `${OUT}/race.gif`]);
  }
  rmSync(tmp, { recursive: true, force: true });
}
await browser.close();
