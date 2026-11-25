import { describe, expect, it } from 'vitest';
import { existsSync } from 'node:fs';
import config, { coopCoep } from '../../vite.config';

describe('COOP/COEP', () => {
  it('dev middleware sets isolation headers', () => {
    const headers: Record<string, string> = {};
    let called = false;
    const plugin = coopCoep();
    const hook = plugin.configureServer;
    if (typeof hook !== 'function') throw new Error('expected configureServer function');
    hook({
      middlewares: {
        use(fn: (req: unknown, res: { setHeader: (k: string, v: string) => void }, next: () => void) => void) {
          fn({}, { setHeader: (k, v) => { headers[k] = v; } }, () => { called = true; });
        },
      },
    } as never);
    expect(headers['Cross-Origin-Opener-Policy']).toBe('same-origin');
    expect(headers['Cross-Origin-Embedder-Policy']).toBe('require-corp');
    expect(called).toBe(true);
  });

  it('preview middleware sets the same headers', () => {
    const headers: Record<string, string> = {};
    const hook = coopCoep().configurePreviewServer;
    if (typeof hook !== 'function') throw new Error('expected configurePreviewServer function');
    hook({
      middlewares: {
        use(fn: (req: unknown, res: { setHeader: (k: string, v: string) => void }, next: () => void) => void) {
          fn({}, { setHeader: (k, v) => { headers[k] = v; } }, () => undefined);
        },
      },
    } as never);
    expect(headers).toEqual({ 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' });
  });

  it('server and preview config carry the headers, and static hosting has the service worker shim', () => {
    const cfg = config as { server?: { headers?: Record<string, string> }; preview?: { headers?: Record<string, string> } };
    for (const block of [cfg.server, cfg.preview]) {
      expect(block?.headers?.['Cross-Origin-Opener-Policy']).toBe('same-origin');
      expect(block?.headers?.['Cross-Origin-Embedder-Policy']).toBe('require-corp');
    }
    // GitHub Pages cannot set headers; public/coi-serviceworker.js re-serves the page with them.
    expect(existsSync('public/coi-serviceworker.js')).toBe(true);
  });
});
