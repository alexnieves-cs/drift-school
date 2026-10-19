import { trainRecipe, type RecipeResult } from '../rl/recipe';

const cache = new Map<string, Promise<RecipeResult>>();

/**
 * One trained trainer shared by every eval section so training time is paid once.
 * `quick` skips the PPO phase (imitation only) to shorten smoke runs.
 */
export function sharedTrained(quick = false): Promise<RecipeResult> {
  const key = quick ? 'quick' : 'full';
  let p = cache.get(key);
  if (!p) {
    p = trainRecipe(quick ? { ppoUpdates: 0 } : {});
    cache.set(key, p);
  }
  return p;
}
