import type { GenStat } from '../shared/types';

export class Leaderboard {
  readonly rows: GenStat[] = [];
  private body: HTMLElement;
  private onPick: (gen: number) => void;

  constructor(parent: HTMLElement, onPick: (gen: number) => void) {
    this.onPick = onPick;
    const table = document.createElement('table');
    table.className = 'board';
    table.innerHTML = `<thead><tr><th>gen</th><th>best</th><th>mean</th><th>done</th><th>crash</th><th>vmax</th></tr></thead><tbody></tbody>`;
    parent.appendChild(table);
    this.body = table.querySelector('tbody')!;
  }

  add(stat: GenStat): void {
    this.rows.push(stat);
    const tr = document.createElement('tr');
    tr.dataset.gen = String(stat.gen);
    const fmt = (v: number | null) => (v == null ? '—' : v.toFixed(2));
    tr.innerHTML = `<td>${stat.gen}</td><td>${fmt(stat.bestLap)}</td><td>${fmt(stat.meanLap)}</td><td>${(stat.completion * 100).toFixed(0)}%</td><td>${(stat.crashRate * 100).toFixed(0)}%</td><td>${stat.topSpeed.toFixed(1)}</td>`;
    tr.addEventListener('click', () => this.onPick(stat.gen));
    this.body.prepend(tr);
    const extra = this.body.querySelectorAll('tr');
    if (extra.length > 12) extra[extra.length - 1]?.remove();
  }
}
