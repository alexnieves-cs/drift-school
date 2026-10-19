import type { GenStat } from '../shared/types';

/** Rows kept in the DOM. The model (`rows`) always holds every generation. */
export const MAX_DOM_ROWS = 400;

function fmt(v: number | null): string {
  return v == null ? '—' : v.toFixed(2);
}

/**
 * One row per PPO update, newest first. `rows` is the full history, the table body is capped at
 * MAX_DOM_ROWS so a long session does not grow the DOM without bound. `data-rows` (on the table and
 * its parent) carries the total.
 */
export class Leaderboard {
  readonly rows: GenStat[] = [];
  private parent: HTMLElement;
  private table: HTMLTableElement;
  private body: HTMLElement;
  private onPick: (gen: number) => void;

  constructor(parent: HTMLElement, onPick: (gen: number) => void) {
    this.onPick = onPick;
    this.parent = parent;
    parent.dataset.rows = '0';
    this.table = document.createElement('table');
    this.table.className = 'board';
    this.table.dataset.rows = '0';
    this.table.innerHTML = `<thead><tr><th>gen</th><th>best</th><th>mean</th><th>done</th><th>crash</th><th>vmax</th></tr></thead><tbody></tbody>`;
    parent.appendChild(this.table);
    this.body = this.table.querySelector('tbody')!;
    // One listener for every row instead of one closure per row.
    this.body.addEventListener('click', (ev) => this.pick(ev.target as HTMLElement));
    this.body.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        this.pick(ev.target as HTMLElement);
      }
    });
  }

  private pick(el: HTMLElement): void {
    const tr = el.closest('tr');
    if (tr?.dataset.gen) this.onPick(Number(tr.dataset.gen));
  }

  add(stat: GenStat): void {
    this.rows.push(stat);
    const tr = document.createElement('tr');
    tr.dataset.gen = String(stat.gen);
    tr.tabIndex = 0;
    tr.setAttribute('role', 'button');
    tr.title = `replay generation ${stat.gen}`;
    tr.innerHTML = `<td>${stat.gen}</td><td>${fmt(stat.bestLap)}</td><td>${fmt(stat.meanLap)}</td><td>${stat.episodes ? `${(stat.completion * 100).toFixed(0)}%` : '—'}</td><td>${stat.episodes ? `${(stat.crashRate * 100).toFixed(0)}%` : '—'}</td><td>${stat.episodes ? stat.topSpeed.toFixed(1) : '—'}</td>`;
    this.body.prepend(tr);
    if (this.body.childElementCount > MAX_DOM_ROWS) this.body.lastElementChild?.remove();
    this.table.dataset.rows = String(this.rows.length);
    this.parent.dataset.rows = String(this.rows.length);
  }

  /** Mark the row being replayed. */
  mark(gen: number | null): void {
    this.body.querySelectorAll('tr.replaying').forEach((r) => r.classList.remove('replaying'));
    if (gen != null) this.body.querySelector(`tr[data-gen="${gen}"]`)?.classList.add('replaying');
  }
}
