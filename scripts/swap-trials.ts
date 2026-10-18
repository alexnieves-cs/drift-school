import { trainBaseRecipe } from '../src/rl/recipe';
import { Trainer } from '../src/rl/trainer';
import { runFineTune, snapshot } from '../src/eval/hotswap';
import { generateTrack } from '../src/track/generate';
import { processStroke, sCurveStroke, triangleStroke } from '../src/editor/pipeline';
import { quantizePoints } from '../src/editor/share';
import { toFloat } from '../src/editor/stroke';
import { buildTrackFromPoints } from '../src/track/build';
import { pct } from '../src/eval/harness';

const trials = Number(process.argv[2] ?? 3);
const windowS = Number(process.argv[3] ?? 120);
const flavor = process.argv[4] ?? 'recipe';
const flat = quantizePoints(toFloat(processStroke(sCurveStroke(), 1, 1.05).points));
const user = buildTrackFromPoints(flat, { halfWidth: 1, mu: 1.05, source: 'user' });
const drawn = processStroke(triangleStroke(), 1, 1.05).track!;
const pool = [1, 2, 3, 4].map((seed) => generateTrack({ seed, stage: 0 }));
const held = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116].map((seed) => generateTrack({ seed, stage: 0 }));
for (let k = 0; k < trials; k++) {
  let base: Trainer;
  if (flavor === 'recipe') base = (await trainBaseRecipe()).trainer;
  else {
    base = new Trainer({ n: 8, horizon: 48, epochs: 2, minibatch: 256 });
    await base.init(pool);
    await base.imitate(4);
  }
  const before = snapshot((o, n) => base.policy.actMean(o, n), user, drawn, held, pool);
  const r = await runFineTune(base, pool, user, drawn, held, 'mixed', windowS, () => undefined);
  const a = r.after;
  console.log(
    `trial ${k} (${flavor}): before user ${pct(before.user)} drDR ${pct(before.userDr)} held ${pct(before.heldProc)} pool ${pct(before.poolProc)} drawn ${pct(before.heldDrawn)} | after user ${pct(a.user)} userDR ${pct(a.userDr)} held ${pct(a.heldProc)} pool ${pct(a.poolProc)} drawn ${pct(a.heldDrawn)} | latency ${((r.report.latencyMs ?? 0) / 1000).toFixed(1)}s zeroShot ${r.report.zeroShot ? pct(r.report.zeroShot.completion) : '-'}/${r.report.zeroShot?.episodes} unchanged ${r.report.weightsUnchanged} 80%@${r.secondsTo80?.toFixed(0)}s probes ${r.probes}`,
  );
}
process.exit(0);
