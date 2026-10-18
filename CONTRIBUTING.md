# Contributing

Thanks for looking. This is a personal project, and small, focused pull requests are the easiest to review.

## Setup

Node 22 and npm.

```bash
npm ci
npm run dev       # http://127.0.0.1:5173, with COOP/COEP headers
```

## Before you open a pull request

```bash
npm run lint
npm test          # unit tests and the perf budget checks
npm run build     # typecheck and production bundle
```

`npm run test:e2e` runs the Playwright specs. It needs a Chromium that can create a WebGL context. If Playwright cannot download its own browser, point `PW_CHROMIUM_PATH` at a system Chromium.

`npm run eval` prints the acceptance table from SPEC section 3 (`--quick` for a short run, `--only=stage0,matrix,hotswap,fps` for a subset). If your change touches the sim, the reward, the trainer or the renderer, include the table's before and after in the pull request.

## Ground rules

- **Keep the sim headless.** `src/sim`, `src/track`, `src/rl` and the env workers must not import Three.js, uPlot, `ui/` or `viz/`, and must not touch DOM globals. ESLint fails the build if they do.
- **No `Math.random` in the sim.** Take an RNG from `src/shared/rng.ts` and `fork` it by label. Same seed, same result is a tested property.
- **`src/shared` holds the contracts.** Changing `TrackSpec`, `CarParams`, the observation layout or the worker messages touches everything. Update `src/sim/OBS_LAYOUT.md` with any observation change, and say why in the pull request. A new observation layout invalidates saved checkpoints.
- **Numbers come from scripts.** If a README or docs figure changed, regenerate it with `npm run eval`, `npm run bench` or `scripts/record-demos.mjs` and paste the output. Do not round a result up, and mark anything that was not measured.
- **Tests with the change.** Fixtures for new track-validation reasons go in `tests/fixtures`. Physics changes need a test that would have caught the regression.

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what each directory owns and how data moves between them.

## Re-recording the demo media

The GIFs and videos in `docs/media` are made by `scripts/record-demos.mjs`: Playwright screenshots encoded with ffmpeg. Start `npm run dev`, then:

```bash
node scripts/record-demos.mjs hairpin      # real vehicle model, dry vs ice
RACE_MINUTES=4 node scripts/record-demos.mjs race
DRAW_MINUTES=3 node scripts/record-demos.mjs draw
```

Set `CHROME_PATH` to use a specific Chromium. Files under 5 MB are written to `docs/media`. Larger ones go to `release-assets/` (git-ignored) and belong in a GitHub Release.

## License

By contributing you agree that your contribution is released under the [MIT license](LICENSE).
