import type { GenStat } from '../shared/types';

export interface EpRecord {
  lap: boolean;
  lapTime?: number;
  crash: boolean;
  ret: number;
  trackId: string;
  speed: number;
}

export function aggregateGen(gen: number, eps: EpRecord[], trackId: string): GenStat {
  if (!eps.length) {
    return { gen, bestLap: null, meanLap: null, completion: 0, crashRate: 0, topSpeed: 0, meanReturn: 0, trackId };
  }
  const laps = eps.map((e) => e.lapTime).filter((t): t is number => typeof t === 'number' && t > 0);
  const bestLap = laps.length ? Math.min(...laps) : null;
  const meanLap = laps.length ? laps.reduce((a, b) => a + b, 0) / laps.length : null;
  return {
    gen,
    bestLap,
    meanLap,
    completion: eps.filter((e) => e.lap).length / eps.length,
    crashRate: eps.filter((e) => e.crash).length / eps.length,
    topSpeed: eps.reduce((m, e) => Math.max(m, e.speed), 0),
    meanReturn: eps.reduce((s, e) => s + e.ret, 0) / eps.length,
    trackId,
  };
}
