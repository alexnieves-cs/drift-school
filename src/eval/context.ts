import { trainBaseRecipe } from '../rl/recipe';
import type { Trainer } from '../rl/trainer';

let cached: Promise<{ trainer: Trainer; seconds: number }> | null = null;

/** One trained trainer shared by every eval section so training time is paid once. */
export function sharedTrained(): Promise<{ trainer: Trainer; seconds: number }> {
  cached ??= trainBaseRecipe();
  return cached;
}
