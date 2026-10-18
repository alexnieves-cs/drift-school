# Requests from the RL stream to `src/sim/**` (physics worker)

Nothing here blocks the RL work. These would make training faster or cleaner.

1. **Per-slot reset with a new config.** `VecEnv.step` re-places a finished slot on the same
   track with the same `EpisodeConfig`, so DR is not resampled per episode and a finished slot
   cannot move to another track. The recipe works around it by calling `reset` with fresh
   scenarios at the start of every rollout (`Trainer.resetEnvs`). A `resetSlot(i, track, cfg)`
   would let PPO keep long, continuous rollouts and still redraw DR/surface per episode.
2. **`sampleAtS` is a linear scan** over the track samples, and `baselineActions` calls it twice
   per car per step. Imitation collects about 100k expert labels per recipe run; a cached index
   or binary search there is the cheapest speed-up for the recipe (env steps are ~3k/s/core).
3. **Baseline contract.** The recipe distils `purePursuit` (src/track/baseline.ts). If the
   baseline or tire parameters are retuned, re-run `npx vite-node scripts/rl-runs.ts -- --runs=3`;
   the recipe needs the baseline to finish stage 0-3 tracks (it does at ~100% today, across all
   surfaces and DR levels measured here, see NOTES/rl.md).
