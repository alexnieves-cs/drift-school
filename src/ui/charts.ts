import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { GenStat } from '../shared/types';

export class MetricCharts {
  private plots: uPlot[] = [];
  private xs: number[] = [];
  private series: number[][] = [[], [], [], [], []];

  constructor(parent: HTMLElement) {
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
    this.plots.forEach((plot, i) => plot.setData([this.xs, this.series[i]!]));
  }

  resize(width: number): void {
    this.plots.forEach((plot) => plot.setSize({ width, height: 92 }));
  }
}
