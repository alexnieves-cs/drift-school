/** How often does the legacy ray rule disagree with the true nearest hit? `npx vite-node scripts/ray-bug.ts` */
import { RAY_COUNT } from '../src/shared/constants';
import { makeRng } from '../src/shared/rng';
import { RayField, SENSOR_FLAGS } from '../src/sim/sensors';
import { generateTrack } from '../src/track/generate';

for (const stage of [0, 1, 2, 3, 4]) {
  let rays = 0;
  let wrong = 0;
  let sumErr = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const track = generateTrack({ seed, stage });
    const rf = new RayField();
    rf.setTrack(track);
    const rng = makeRng(seed, 'raybug');
    const a = new Float32Array(RAY_COUNT);
    const b = new Float32Array(RAY_COUNT);
    for (let n = 0; n < 300; n++) {
      const s = track.samples[rng.int(track.samples.length)]!;
      const lat = rng.range(-0.8, 0.8) * s.halfWidth;
      const x = s.x + s.nx * lat;
      const z = s.z + s.nz * lat;
      const yaw = Math.atan2(s.tz, s.tx) + rng.range(-0.5, 0.5);
      SENSOR_FLAGS.nearestRay = false;
      rf.cast(x, z, yaw, a, 0, null, 0);
      SENSOR_FLAGS.nearestRay = true;
      rf.cast(x, z, yaw, b, 0, null, 0);
      for (let r = 0; r < RAY_COUNT; r++) {
        rays++;
        if (a[r] !== b[r]) {
          wrong++;
          sumErr += Math.abs(a[r]! - b[r]!);
        }
      }
    }
  }
  SENSOR_FLAGS.nearestRay = false;
  console.log(`stage ${stage}: ${((wrong / rays) * 100).toFixed(1)}% of rays differ, mean |error| when wrong ${(sumErr / Math.max(1, wrong)).toFixed(2)} (fraction of 6 m)`);
}
