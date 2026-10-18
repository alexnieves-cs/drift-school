import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { GenStat } from '../shared/types';

/**
 * Streaming metric curves. `push` only records the sample; `flush` redraws and is called from
 * the UI tick (a few Hz), so a burst of updates costs one redraw and nothing is drawn while the
 * panel is hidden.
 */
export class MetricCharts {
  private plots: uPlot[] = [];
  private xs: number[] = [];
  private series: number[][] = [[], [], [], [], []];
  private dirty = false;
  private parent: HTMLElement;
  /** Number of uPlot redraws so far, for tests and profiling. */
  redraws = 0;

  constructor(parent: HTMLElement) {
    this.parent = parent;
    const labels = ['return', 'completion', 'entropy', 'steps/s', 'value loss'];
    const colors = ['#ffb020', '#3dd68c', '#5b8def', '#ff5d73', '#c084fc'];
    labels.forEach((label, i) => {
      const host = document.createElement('div');
      host.className = 'chart';
      parent.appendChild(host);
      const plot = new uPlot(
        {
          width: 280,
          height: 92,
          legend: { show: false },
          scales: { x: { time: false } },
          axes: [
            { stroke: '#8b93a7', grid: { stroke: '#243044' }, size: 28 },
            { stroke: '#8b93a7', grid: { stroke: '#243044' }, size: 36 },
          ],
          series: [{}, { label, stroke: colors[i], width: 1.6 }],
        },
        [[], []],
        host,
      );
      this.plots.push(plot);
    });
  }

  push(stat: GenStat): void {
    this.xs.push(stat.gen);
    this.series[0]!.push(stat.meanReturn);
    this.series[1]!.push(stat.completion);
    this.series[2]!.push(stat.entropy ?? 0);
    this.series[3]!.push(stat.stepsPerSec ?? 0);
    this.series[4]!.push(stat.valueLoss ?? 0);
    this.dirty = true;
  }

  get samples(): number {
    return this.xs.length;
  }

  /** Redraw if there is new data and the panel is visible. Returns true when it drew. */
  flush(): boolean {
    if (!this.dirty) return false;
    if (this.parent.offsetParent === null) return false;
    this.dirty = false;
    this.plots.forEach((plot, i) => plot.setData([this.xs, this.series[i]!]));
    this.redraws++;
    return true;
  }

  resize(width: number): void {
    this.plots.forEach((plot) => plot.setSize({ width, height: 92 }));
  }
}
