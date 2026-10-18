import type { TrackSpec } from '../shared/types';

interface Layout {
  key: string;
  /** Canvas-space transform. */
  ox: number;
  oz: number;
  sc: number;
  /** One polyline per metre of arc length, so a bin is one stroke instead of ten segments. */
  bins: { bin: number; path: Path2D }[];
}

/** Centerline colored by the latest sampled speed, plus crash dots. */
export class Heatmap {
  private canvas: HTMLCanvasElement;
  private speeds = new Map<number, number>();
  private crashes: { x: number; z: number }[] = [];
  private layout: Layout | null = null;
  private dirty = true;
  /** Full repaints so far, for tests and profiling. */
  draws = 0;

  constructor(parent: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'heatmap';
    this.canvas.width = 280;
    this.canvas.height = 140;
    parent.appendChild(this.canvas);
  }

  /** Record one observation. Cheap; the repaint happens in flush(). */
  note(track: TrackSpec, s: number, speed: number, crash: boolean, x: number, z: number): void {
    void track;
    const bin = Math.round(s);
    const prev = this.speeds.get(bin) ?? speed;
    this.speeds.set(bin, prev * 0.8 + speed * 0.2);
    if (crash) {
      this.crashes.push({ x, z });
      if (this.crashes.length > 80) this.crashes.shift();
    }
    this.dirty = true;
  }

  /** Repaint if anything changed since the last repaint. Returns true when it drew. */
  flush(track: TrackSpec): boolean {
    if (!this.dirty && this.layout?.key === keyOf(track)) return false;
    this.draw(track);
    return true;
  }

  private ensureLayout(track: TrackSpec): Layout {
    const key = keyOf(track);
    if (this.layout?.key === key) return this.layout;
    const w = this.canvas.width;
    const h = this.canvas.height;
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (const s of track.samples) {
      minX = Math.min(minX, s.x);
      maxX = Math.max(maxX, s.x);
      minZ = Math.min(minZ, s.z);
      maxZ = Math.max(maxZ, s.z);
    }
    const sc = Math.min((w - 16) / Math.max(1, maxX - minX), (h - 16) / Math.max(1, maxZ - minZ));
    const ox = 8 - minX * sc;
    const oz = 8 - minZ * sc;
    const bins: Layout['bins'] = [];
    let cur: { bin: number; path: Path2D } | null = null;
    for (let i = 1; i < track.samples.length; i++) {
      const a = track.samples[i - 1]!;
      const b = track.samples[i]!;
      const bin = Math.round(b.s);
      if (!cur || cur.bin !== bin) {
        cur = { bin, path: new Path2D() };
        cur.path.moveTo(ox + a.x * sc, oz + a.z * sc);
        bins.push(cur);
      }
      cur.path.lineTo(ox + b.x * sc, oz + b.z * sc);
    }
    this.layout = { key, ox, oz, sc, bins };
    // The speed map belongs to one track; drop it when the track changes.
    this.speeds.clear();
    this.crashes.length = 0;
    return this.layout;
  }

  draw(track: TrackSpec): void {
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    const lay = this.ensureLayout(track);
    this.dirty = false;
    this.draws++;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    for (const { bin, path } of lay.bins) {
      const v = this.speeds.get(bin) ?? 0;
      const t = Math.max(0, Math.min(1, v / 6));
      ctx.strokeStyle = `rgb(${Math.round(40 + t * 215)}, ${Math.round(180 - t * 80)}, ${Math.round(80 + (1 - t) * 80)})`;
      ctx.stroke(path);
    }
    ctx.fillStyle = '#ff5d73';
    for (const c of this.crashes) ctx.fillRect(lay.ox + c.x * lay.sc - 1.5, lay.oz + c.z * lay.sc - 1.5, 3, 3);
  }
}

function keyOf(track: TrackSpec): string {
  return `${track.id}:${track.samples.length}`;
}
