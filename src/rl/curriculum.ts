import type { RNG } from '../shared/rng';

export interface StageRule {
  stage: number;
  completion: number;
  /** Promote only if mean lap time is below this multiple of tRef. Infinity skips the check. */
  timeFactor: number;
}

export const STAGE_RULES: StageRule[] = [
  { stage: 0, completion: 0.8, timeFactor: Infinity },
  { stage: 1, completion: 0.8, timeFactor: Infinity },
  { stage: 2, completion: 0.75, timeFactor: 1.3 },
  { stage: 3, completion: 0.7, timeFactor: Infinity },
  { stage: 4, completion: 1.01, timeFactor: Infinity },
];

export interface EpOutcome {
  completed: boolean;
  lapTime?: number;
  tRef: number;
}

/**
 * Stage table with a 20% replay of earlier stages.
 * Promotion looks at a sliding window of recent episodes.
 */
export class Curriculum {
  stage = 0;
  private window: EpOutcome[] = [];
  readonly windowSize: number;

  constructor(windowSize = 32) {
    this.windowSize = windowSize;
  }

  note(ep: EpOutcome): void {
    this.window.push(ep);
    if (this.window.length > this.windowSize) this.window.shift();
  }

  stats(): { completion: number; timeOk: boolean } {
    if (!this.window.length) return { completion: 0, timeOk: false };
    const completion = this.window.filter((e) => e.completed).length / this.window.length;
    const rule = STAGE_RULES[this.stage] ?? STAGE_RULES[4]!;
    const timed = this.window.filter((e) => e.completed && e.lapTime && e.tRef > 0);
    const timeOk =
      rule.timeFactor === Infinity ||
      timed.length === 0 ||
      timed.reduce((s, e) => s + (e.lapTime! < e.tRef * rule.timeFactor ? 1 : 0), 0) / timed.length >= 0.5;
    return { completion, timeOk };
  }

  maybePromote(): boolean {
    if (this.stage >= 4) return false;
    if (this.window.length < Math.min(16, this.windowSize)) return false;
    const rule = STAGE_RULES[this.stage]!;
    const { completion, timeOk } = this.stats();
    if (completion >= rule.completion && timeOk) {
      this.stage++;
      this.window = [];
      return true;
    }
    return false;
  }

  /** 80% current stage, 20% uniform over prior stages. */
  sampleStage(rng: RNG): number {
    if (this.stage === 0 || rng.next() > 0.2) return this.stage;
    return rng.int(this.stage);
  }
}
