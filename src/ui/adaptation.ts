import type { FineTuneReport } from '../rl/finetune';

/**
 * "Adaptation" panel (SPEC editor step 4): completion rate and best lap on the user track against
 * learner steps since the swap, plus the zero-shot result (first 64 episodes, no update), the swap
 * latency and the weight-hash check. Data attributes mirror what is printed so tests can read them.
 */
export class AdaptationPanel {
  readonly el: HTMLElement;
  private readonly body: HTMLElement;
  private readonly spark: SVGSVGElement;
  private readonly points: { steps: number; completion: number }[] = [];

  constructor(parent: HTMLElement) {
    this.el = document.createElement('section');
    this.el.className = 'adaptation';
    this.el.dataset.phase = 'idle';
    this.el.hidden = true;
    this.el.innerHTML = `<h3>adaptation</h3><div class="body"></div>
      <svg class="spark" viewBox="0 0 160 36" preserveAspectRatio="none" aria-label="user-track completion vs learner steps"></svg>`;
    this.body = this.el.querySelector('.body') as HTMLElement;
    this.spark = this.el.querySelector('.spark') as SVGSVGElement;
    parent.appendChild(this.el);
  }

  begin(trackId: string): void {
    this.points.length = 0;
    this.el.hidden = false;
    this.el.dataset.phase = 'drain';
    this.el.dataset.track = trackId;
    for (const k of ['latencyMs', 'zeroShot', 'completion', 'bestLap', 'unchanged', 'steps']) delete this.el.dataset[k];
    this.body.textContent = `swapping to ${trackId}`;
    this.spark.innerHTML = '';
  }

  update(r: FineTuneReport): void {
    const d = this.el.dataset;
    d.phase = r.phase;
    d.steps = String(r.steps);
    if (r.latencyMs !== null) d.latencyMs = String(Math.round(r.latencyMs));
    if (r.zeroShot) d.zeroShot = r.zeroShot.completion.toFixed(3);
    if (r.userCompletion !== null) d.completion = r.userCompletion.toFixed(3);
    if (r.userBestLap !== null) d.bestLap = r.userBestLap.toFixed(2);
    if (r.weightsUnchanged !== null) d.unchanged = String(r.weightsUnchanged);
    if (r.phase === 'tune' && r.userCompletion !== null) {
      const last = this.points[this.points.length - 1];
      if (!last || last.steps !== r.steps) this.points.push({ steps: r.steps, completion: r.userCompletion });
    }
    const pct = (x: number) => `${Math.round(x * 100)}%`;
    const lines = [
      `${r.phase === 'drain' ? 'finishing episodes' : r.phase === 'zeroShot' ? 'zero-shot episodes' : 'fine-tuning'}: ${r.envsOnUser}/${r.envTotal} envs on the new track`,
      r.latencyMs !== null ? `swap latency ${(r.latencyMs / 1000).toFixed(1)} s, weights ${r.weightsUnchanged === null ? 'unchanged so far' : r.weightsUnchanged ? 'unchanged' : 'CHANGED'}` : '',
      r.zeroShot ? `zero-shot ${pct(r.zeroShot.completion)} of ${r.zeroShot.episodes} episodes` : '',
      r.phase === 'tune'
        ? `steps ${r.steps} (${r.ppoUpdates} PPO, ${r.imitationBatches} imitation) · completion ${r.userCompletion === null ? '-' : pct(r.userCompletion)} · best lap ${r.userBestLap === null ? '-' : `${r.userBestLap.toFixed(1)} s`}`
        : '',
    ].filter(Boolean);
    this.body.innerHTML = lines.map((l) => `<p>${l}</p>`).join('');
    this.draw();
  }

  private draw(): void {
    if (this.points.length < 2) return;
    const maxStep = this.points[this.points.length - 1]!.steps || 1;
    const path = this.points
      .map((p, i) => `${i ? 'L' : 'M'}${((p.steps / maxStep) * 158 + 1).toFixed(1)},${(34 - p.completion * 32).toFixed(1)}`)
      .join('');
    this.spark.innerHTML = `<path d="${path}" fill="none" stroke="currentColor" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
  }
}
