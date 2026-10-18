import { generateTrack } from '../track/generate';
import { Trainer } from './trainer';

export interface RecipeOpts {
  /** Stage-0 training seeds. Evaluation must use other seeds. */
  seeds?: number[];
  n?: number;
}

/** The recipe the app ships: pure-pursuit imitation warm start, then PPO. */
export async function trainBaseRecipe(opts: RecipeOpts = {}): Promise<{ trainer: Trainer; seconds: number }> {
  const t0 = performance.now();
  const tracks = (opts.seeds ?? [1, 2, 3, 4]).map((seed) => generateTrack({ seed, stage: 0 }));
  const trainer = new Trainer({ n: opts.n ?? 8, horizon: 48, epochs: 2, minibatch: 256 });
  await trainer.init(tracks);
  await trainer.imitate(4);
  await trainer.update();
  return { trainer, seconds: (performance.now() - t0) / 1000 };
}
