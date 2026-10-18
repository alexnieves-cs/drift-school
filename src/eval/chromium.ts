import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from '@playwright/test';

/**
 * Chromium binary to drive. The playwright version pinned in package.json may want a browser
 * revision the machine does not have (browsers come from PLAYWRIGHT_BROWSERS_PATH and are never
 * downloaded here), so fall back to the highest revision that is actually installed.
 * Override with DRIFT_CHROMIUM=/path/to/chrome.
 */
export function findChromium(): string | null {
  const env = process.env.DRIFT_CHROMIUM;
  if (env && existsSync(env)) return env;
  try {
    const p = chromium.executablePath();
    if (p && existsSync(p)) return p;
  } catch {
    /* fall through to the cache scan */
  }
  const base =
    process.env.PLAYWRIGHT_BROWSERS_PATH && process.env.PLAYWRIGHT_BROWSERS_PATH !== '0'
      ? process.env.PLAYWRIGHT_BROWSERS_PATH
      : join(process.env.HOME ?? '', '.cache', 'ms-playwright');
  if (!existsSync(base)) return null;
  const candidates: { rev: number; path: string }[] = [];
  for (const dir of readdirSync(base)) {
    const m = /^(chromium_headless_shell|chromium)-(\d+)$/.exec(dir);
    if (!m) continue;
    const rev = Number(m[2]);
    const rels =
      m[1] === 'chromium_headless_shell'
        ? ['chrome-linux/headless_shell', 'chrome-headless-shell-linux64/chrome-headless-shell']
        : ['chrome-linux/chrome', 'chrome-linux64/chrome'];
    for (const rel of rels) {
      const path = join(base, dir, rel);
      if (existsSync(path)) candidates.push({ rev, path });
    }
  }
  candidates.sort((a, b) => b.rev - a.rev);
  return candidates[0]?.path ?? null;
}

/** Flags that give a WebGL context in headless Chromium, using the GPU when there is one. */
export const CHROMIUM_GL_ARGS = ['--enable-webgl', '--ignore-gpu-blocklist', '--use-gl=angle', '--enable-unsafe-swiftshader'];

const HAS_PROC = existsSync('/proc/self/stat');

/**
 * CPU time (ms) used so far by a process and everything it spawned, from /proc. Linux only, null
 * elsewhere. Wall-clock fps moves with whatever else is running; CPU time per frame moves much less.
 */
export function treeCpuMs(rootPid: number): number | null {
  if (!HAS_PROC || rootPid <= 0) return null;
  const parent = new Map<number, number>();
  const cpu = new Map<number, number>();
  for (const d of readdirSync('/proc')) {
    if (!/^\d+$/.test(d)) continue;
    try {
      const st = readFileSync(`/proc/${d}/stat`, 'utf8');
      const f = st.slice(st.lastIndexOf(')') + 2).split(' ');
      parent.set(Number(d), Number(f[1]));
      cpu.set(Number(d), Number(f[11]) + Number(f[12]));
    } catch {
      /* the process exited while scanning */
    }
  }
  const inTree = new Set([rootPid]);
  for (let grew = true; grew; ) {
    grew = false;
    for (const [pid, ppid] of parent) {
      if (!inTree.has(pid) && inTree.has(ppid)) {
        inTree.add(pid);
        grew = true;
      }
    }
  }
  let ticks = 0;
  for (const pid of inTree) ticks += cpu.get(pid) ?? 0;
  return ticks * 10; // USER_HZ is 100 on Linux
}

/** Pid of the browser process that was started with this --user-data-dir, or 0. */
export function findBrowserPid(userDataDir: string): number {
  if (!HAS_PROC) return 0;
  for (const d of readdirSync('/proc')) {
    if (!/^\d+$/.test(d)) continue;
    try {
      const cmd = readFileSync(`/proc/${d}/cmdline`, 'utf8');
      if (cmd.includes(`--user-data-dir=${userDataDir}`) && !cmd.includes('--type=')) return Number(d);
    } catch {
      /* the process exited while scanning */
    }
  }
  return 0;
}
