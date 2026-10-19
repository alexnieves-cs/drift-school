import type { TrackSpec, Validation } from '../shared/types';
import { buildTrackFromPoints, validateTrack } from '../track/build';
import { dist, fromFloat, smoothStroke, toFloat, trimClosingTail, type V2 } from './stroke';

/** Make closure explicit: the validator and the build read a repeated first point as "closed". */
export function closeLoop(points: V2[], closed: boolean): V2[] {
  if (!closed || points.length < 3) return points;
  return [...points, { ...points[0]! }];
}

export interface EditorOpts {
  width: number;
  height: number;
  /** World units per pixel. */
  scale: number;
  onChange?: (state: EditorState) => void;
}

export interface EditorState {
  points: V2[];
  halfWidth: number;
  mu: number;
  validation: Validation;
  track: TrackSpec | null;
  ms: number;
}

export class TrackEditor {
  points: V2[] = [];
  halfWidth = 1;
  mu = 1.05;
  mode: 'stroke' | 'place' = 'stroke';
  drawing = false;
  private raw: V2[] = [];
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private opts: EditorOpts;
  drag = -1;

  constructor(canvas: HTMLCanvasElement, opts: EditorOpts) {
    this.canvas = canvas;
    this.opts = opts;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2d context missing');
    this.ctx = ctx;
    canvas.addEventListener('pointerdown', (e) => this.onDown(e));
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerup', (e) => {
      if (e.isPrimary) this.onUp();
    });
    // A cancelled touch or lost capture ends the stroke instead of leaving the editor mid-draw.
    canvas.addEventListener('pointercancel', () => this.onUp());
    canvas.addEventListener('lostpointercapture', () => this.onUp());
  }

  setSize(w: number, h: number, scale: number): void {
    this.opts.width = w;
    this.opts.height = h;
    this.opts.scale = scale;
    this.canvas.width = w;
    this.canvas.height = h;
  }

  world(ev: { clientX: number; clientY: number }): V2 {
    const r = this.canvas.getBoundingClientRect();
    const x = (ev.clientX - r.left - this.opts.width / 2) * this.opts.scale;
    const z = (ev.clientY - r.top - this.opts.height / 2) * this.opts.scale;
    return { x, z };
  }

  private onDown(ev: PointerEvent): void {
    if (!ev.isPrimary) return;
    try {
      this.canvas.setPointerCapture(ev.pointerId);
    } catch {
      // Synthetic events have no active pointer to capture.
    }
    const p = this.world(ev);
    if (this.mode === 'place') {
      this.points.push(p);
      this.drawing = false;
      this.commit();
      return;
    }
    const hit = this.hitHandle(p);
    if (hit >= 0) {
      this.drag = hit;
      return;
    }
    this.drawing = true;
    this.raw = [p];
    this.points = [p];
  }

  private onMove(ev: PointerEvent): void {
    if (!ev.isPrimary) return;
    const p = this.world(ev);
    if (this.drag >= 0) {
      this.points[this.drag] = p;
      this.commit();
      return;
    }
    if (!this.drawing) return;
    const last = this.raw[this.raw.length - 1];
    if (!last || dist(last, p) > 0.15) {
      this.raw.push(p);
      this.points = this.raw.slice();
      this.draw();
    }
  }

  private onUp(): void {
    if (this.drag >= 0) {
      this.drag = -1;
      this.commit();
      return;
    }
    if (!this.drawing) return;
    this.drawing = false;
    if (this.raw.length < 3) return;
    const a = this.raw[0]!;
    const b = this.raw[this.raw.length - 1]!;
    const closed = dist(a, b) < 1;
    const candidates = [smoothStroke(this.raw, closed, this.halfWidth), closed ? trimClosingTail(this.raw) : this.raw.slice()];
    this.points = closeLoop(candidates[0]!, closed);
    for (const candidate of candidates) {
      const closedPts = closeLoop(candidate, closed);
      if (validateTrack(toFloat(closedPts), this.halfWidth).ok) {
        this.points = closedPts;
        break;
      }
      this.points = closedPts;
    }
    this.commit();
  }

  hitHandle(p: V2): number {
    let best = -1;
    let bd = 0.45;
    this.points.forEach((q, i) => {
      const d = dist(p, q);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  }

  load(points: ArrayLike<number>, halfWidth = this.halfWidth, mu = this.mu): EditorState {
    this.points = fromFloat(points);
    this.halfWidth = halfWidth;
    this.mu = mu;
    return this.commit();
  }

  commit(): EditorState {
    const t0 = performance.now();
    const flat = toFloat(this.points);
    const validation = this.points.length >= 3 ? validateTrack(flat, this.halfWidth) : { ok: false, issues: [{ reason: 'open' as const, at: [0, 0] as [number, number] }] };
    let track: TrackSpec | null = null;
    // Only a valid drawing becomes a track, so "Train on this" can never receive a broken one.
    if (validation.ok && this.points.length >= 4) {
      try {
        track = buildTrackFromPoints(flat, { halfWidth: this.halfWidth, mu: this.mu, source: 'user' });
      } catch {
        track = null;
      }
    }
    const state: EditorState = { points: this.points, halfWidth: this.halfWidth, mu: this.mu, validation, track, ms: performance.now() - t0 };
    this.draw(state);
    this.opts.onChange?.(state);
    return state;
  }

  draw(state?: EditorState): void {
    const ctx = this.ctx;
    const { width, height, scale } = this.opts;
    ctx.clearRect(0, 0, width, height);
    const toX = (x: number) => width / 2 + x / scale;
    const toY = (z: number) => height / 2 + z / scale;
    ctx.lineWidth = 2;
    ctx.strokeStyle = state?.validation.ok === false ? '#ff5d5d' : '#ffb020';
    ctx.beginPath();
    this.points.forEach((p, i) => {
      const x = toX(p.x);
      const y = toY(p.z);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    this.points.forEach((p) => {
      ctx.fillStyle = '#efeae2';
      ctx.beginPath();
      ctx.arc(toX(p.x), toY(p.z), 4, 0, Math.PI * 2);
      ctx.fill();
    });
    if (state && !state.validation.ok) {
      ctx.fillStyle = '#ff5d5d';
      ctx.font = '12px ui-monospace, monospace';
      state.validation.issues.slice(0, 3).forEach((iss, i) => {
        ctx.fillText(`${iss.reason} @ ${iss.at[0].toFixed(1)},${iss.at[1].toFixed(1)}`, 12, 18 + i * 16);
      });
    }
  }
}
