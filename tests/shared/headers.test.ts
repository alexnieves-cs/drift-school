import { describe, expect, it } from 'vitest';
import { coopCoep } from '../../vite.config';

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
});
