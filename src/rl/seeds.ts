/**
 * Seed bookkeeping. Training, checkpoint selection and acceptance evaluation use disjoint track seeds.
 * `trainRecipe` refuses any training seed in the reserved ranges.
 */

/** M2 stage-0 acceptance seeds (src/eval/stage0.ts). */
export const STAGE0_HELD_OUT = Array.from({ length: 16 }, (_, i) => 101 + i);

/** M5 matrix seeds (src/eval/matrix.ts): 12 seeds, used at every stage the matrix reports. */
export const MATRIX_SEEDS = Array.from({ length: 12 }, (_, i) => 201 + i);

/** Used only inside the recipe to accept or revert PPO updates and to log phase quality. */
export const VALIDATION_SEEDS = Array.from({ length: 6 }, (_, i) => 301 + i);

/** Default training seeds. */
export const DEFAULT_TRAIN_SEEDS = Array.from({ length: 24 }, (_, i) => 1 + i);

export function reservedSeeds(): Set<number> {
  return new Set([...STAGE0_HELD_OUT, ...MATRIX_SEEDS, ...VALIDATION_SEEDS]);
}

export function assertTrainSeeds(seeds: number[]): void {
  const reserved = reservedSeeds();
  const bad = seeds.filter((s) => reserved.has(s));
  if (bad.length) throw new Error(`training seeds overlap evaluation/validation seeds: ${bad.join(', ')}`);
}
