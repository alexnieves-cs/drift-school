/** Cost of building a track's wall grid. `npx vite-node scripts/grid-cost.ts` */
import { buildWallGrid } from '../src/sim/wallGrid';
import { generateTrack } from '../src/track/generate';

for (const stage of [0, 2, 4]) {
  const tracks = [1, 2, 3, 4, 5, 6, 7, 8].map((seed) => generateTrack({ seed, stage }));
  buildWallGrid(tracks[0]!.walls);
  const t0 = performance.now();
  let bytes = 0;
  let segs = 0;
  for (const t of tracks) {
    const g = buildWallGrid(t.walls);
    bytes += g.seg.byteLength + g.rayOff.byteLength + g.rayIds.byteLength + g.nearOff.byteLength + g.nearIds.byteLength + g.stamp.byteLength;
    segs += g.nSeg;
  }
  const ms = (performance.now() - t0) / tracks.length;
  console.log(`stage ${stage}: ${ms.toFixed(2)} ms/track, ${(bytes / tracks.length / 1024).toFixed(0)} KiB/track, ${(segs / tracks.length).toFixed(0)} segments`);
}
