#!/usr/bin/env node
// Headless demo capture. Frames come from Playwright screenshots, encoded with ffmpeg.
//
//   npm run dev                 (in another shell)
//   node scripts/record-demos.mjs [scene ...]
//
// Scenes: hairpin   real vehicle, pure-pursuit baseline, dry then ice, POV and chase
//         race      worker training with 64 cars, timelapse, writes race-run.json
//         draw      draw a hairpin with the mouse, validate, "train on this", timelapse
// Env:    BASE_URL (default http://127.0.0.1:5173), CHROME_PATH or PW_CHROMIUM_PATH (a Chromium binary),
//         RACE_MINUTES (default 4), DRAW_MINUTES (default 3)
//
// Files under 5 MB go to docs/media. Larger ones go to release-assets/ for a Release.
import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync, statSync, writeFileSync, renameSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const BASE = process.env.BASE_URL ?? 'http://127.0.0.1:5173';
const OUT = 'docs/media';
const BIG = 'release-assets';
const LIMIT = 4.5 * 1024 * 1024;
const W = 960;
const H = 540;

mkdirSync(OUT, { recursive: true });
mkdirSync(BIG, { recursive: true });

async function launch(width = W, height = H) {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || process.env.PW_CHROMIUM_PATH || undefined,
    args: ['--enable-webgl', '--use-gl=angle', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width, height } });
  page.on('pageerror', (e) => console.warn('pageerror:', e.message.slice(0, 200)));
  return { browser, page };
}

function encode(dir, name, fps, { gifWidth = 720, gifFps = 10 } = {}) {
  const mp4 = join(dir, `${name}.mp4`);
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', join(dir, '%05d.png'),
    '-vf', 'scale=1024:-2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '30', '-movflags', '+faststart', mp4]);
  const gif = join(dir, `${name}.gif`);
  const step = Math.max(1, Math.round(fps / gifFps));
  for (let width = gifWidth; width >= 320; width -= 80) {
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', join(dir, '%05d.png'),
      '-vf', `select='not(mod(n,${step}))',setpts=N/(${fps / step}*TB),scale=${width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4`,
      '-r', String(fps / step), gif]);
    if (statSync(gif).size < LIMIT) break;
  }
  for (const f of [mp4, gif]) {
    const size = statSync(f).size;
    const dest = join(size < LIMIT ? OUT : BIG, f.split('/').pop());
    renameSync(f, dest);
    console.log(`${dest}  ${(size / 1024).toFixed(0)} KB${size < LIMIT ? '' : '  (over 5 MB, attach to a Release)'}`);
  }
}

function freshDir(name) {
  const dir = join(tmpdir(), `drift-frames-${name}`);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  return dir;
}

async function hairpin() {
  for (const cam of ['pov', 'chase']) {
    const { browser, page } = await launch();
    await page.goto(`${BASE}/dev/showcase.html?manual=1&cam=${cam}&mus=1.1,0.4`);
    await page.waitForFunction(() => window.__showcase);
    const dir = freshDir(`hairpin-${cam}`);
    let i = 0;
    for (;;) {
      const done = await page.evaluate(() => window.__showcase.advance(2));
      await page.screenshot({ path: join(dir, String(i++).padStart(5, '0') + '.png') });
      if (done || i > 2500) break;
    }
    const results = await page.evaluate(() => window.__showcase.results);
    writeFileSync(join(OUT, `hairpin-${cam}-results.json`), JSON.stringify(results, null, 2) + '\n');
    console.log(cam, JSON.stringify(results));
    await browser.close();
    encode(dir, `demo-hairpin-${cam}`, 25);
  }
}

async function readBoard(page) {
  return page.evaluate(() => {
    const num = (t) => (t === '—' || t === '' ? null : Number(t.replace('%', '')));
    const pct = (t) => (num(t) == null ? null : num(t) / 100);
    return [...document.querySelectorAll('#board tbody tr')].map((tr) => {
      const c = [...tr.children].map((td) => td.textContent ?? '');
      return { gen: Number(c[0]), best: num(c[1]), mean: num(c[2]), done: pct(c[3]), crash: pct(c[4]), vmax: num(c[5]) };
    });
  });
}

async function race() {
  const minutes = Number(process.env.RACE_MINUTES ?? 4);
  const { browser, page } = await launch(1280, 720);
  await page.goto(`${BASE}/?cars=64`);
  await page.waitForSelector('[data-status="racing"]', { timeout: 120_000 });
  const dir = freshDir('race');
  const rows = new Map();
  const t0 = Date.now();
  let i = 0;
  while (Date.now() - t0 < minutes * 60_000) {
    await page.screenshot({ path: join(dir, String(i++).padStart(5, '0') + '.png') });
    for (const r of await readBoard(page)) rows.set(r.gen, r);
    await page.waitForTimeout(800);
  }
  const wall = (Date.now() - t0) / 1000;
  await browser.close();
  const fps = 12;
  const generations = [...rows.values()].sort((a, b) => a.gen - b.gen);
  writeFileSync(join(OUT, 'race-run.json'), JSON.stringify({ cars: 64, wallSeconds: Math.round(wall), frames: i, playbackFps: fps, timelapse: Number((wall / (i / fps)).toFixed(1)), generations }, null, 2) + '\n');
  console.log(`race: ${i} frames over ${wall.toFixed(0)} s wall, ${generations.length} gens read`);
  encode(dir, 'demo-race', fps);
}

// A hairpin traced by the mouse. Points are in the editor's pixel space (0.04 m per pixel).
function hairpinStroke(box) {
  const m = [
    [0, -5], [4, -5], [8, -5], [11.5, -4.4], [13.8, -2.6], [14.6, 0], [13.8, 2.6], [11.5, 4.4],
    [8, 5], [4, 5], [0, 5], [-4, 5], [-8, 5], [-12, 3.6], [-14, 0], [-12, -3.6], [-8, -5], [-4, -5], [0, -5],
  ];
  return m.map(([x, z]) => ({ x: box.x + box.width / 2 + x / 0.04, y: box.y + box.height / 2 + z / 0.04 }));
}

async function draw() {
  const minutes = Number(process.env.DRAW_MINUTES ?? 3);
  const { browser, page } = await launch(1280, 720);
  await page.goto(`${BASE}/?cars=16`);
  await page.waitForSelector('[data-status="racing"]', { timeout: 120_000 });
  const dir = freshDir('draw');
  let i = 0;
  const shot = async () => page.screenshot({ path: join(dir, String(i++).padStart(5, '0') + '.png') });
  for (let k = 0; k < 6; k++) { await shot(); await page.waitForTimeout(400); }
  await page.click('[data-act="editor"]');
  const box = await page.locator('#stroke').boundingBox();
  const pts = hairpinStroke(box);
  await page.mouse.move(pts[0].x, pts[0].y);
  await page.mouse.down();
  for (let a = 1; a < pts.length; a++) {
    const p = pts[a - 1], q = pts[a];
    for (let s = 1; s <= 6; s++) {
      await page.mouse.move(p.x + ((q.x - p.x) * s) / 6, p.y + ((q.y - p.y) * s) / 6);
    }
    await shot();
  }
  await page.mouse.up();
  const verdict = await page.locator('.adapt').first().textContent();
  console.log('editor says:', verdict);
  for (let k = 0; k < 8; k++) { await shot(); await page.waitForTimeout(250); }
  await page.click('[data-act="train"]');
  const t0 = Date.now();
  const adapt = [];
  while (Date.now() - t0 < minutes * 60_000) {
    await shot();
    adapt.push({ t: Math.round((Date.now() - t0) / 1000), adapt: await page.evaluate(() => document.querySelector('.adapt')?.dataset.adapt ?? ''), text: ((await page.locator('.adapt').first().textContent()) ?? '').trim(), track: await page.evaluate(() => document.body.dataset.track ?? ''), board: (await readBoard(page)).slice(0, 1) });
    await page.waitForTimeout(800);
  }
  await browser.close();
  writeFileSync(join(OUT, 'draw-run.json'), JSON.stringify({ editor: verdict, adaptation: adapt }, null, 2) + '\n');
  encode(dir, 'demo-draw-hairpin', 12);
}

const scenes = { hairpin, race, draw };
const wanted = process.argv.slice(2);
for (const name of wanted.length ? wanted : Object.keys(scenes)) {
  if (!scenes[name]) throw new Error(`unknown scene ${name}`);
  console.log(`== ${name}`);
  await scenes[name]();
}
if (existsSync(BIG) && !statSync(BIG).size) rmSync(BIG, { recursive: true });
