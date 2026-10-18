import type { TrackSpec, Validation } from '../shared/types';
import { buildTrackFromPoints, validateTrack } from '../track/build';
import { dist, fromFloat, toFloat, type V2 } from './stroke';
import { processStroke } from './pipeline';

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
    canvas.addEventListener('pointerup', () => this.onUp());
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
    // Keep receiving moves and the release if the pointer leaves the canvas mid-stroke.
    try {
      this.canvas.setPointerCapture(ev.pointerId);
    } catch {
      /* synthetic events have no active pointer to capture */
    }
  }

  private onMove(ev: PointerEvent): void {
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
    // Stroke -> RDP -> Chaikin x2 -> curvature limit -> straight start; see pipeline.ts.
    this.points = processStroke(this.raw, this.halfWidth, this.mu).points;
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
    if (this.points.length >= 4) {
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
