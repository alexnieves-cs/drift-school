import type { TrackSpec } from '../shared/types';

/** Centerline colored by the latest sampled speed, plus crash dots. */
export class Heatmap {
  private canvas: HTMLCanvasElement;
  private speeds = new Map<number, number>();
  private crashes: { x: number; z: number }[] = [];

  constructor(parent: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'heatmap';
    this.canvas.width = 280;
    this.canvas.height = 140;
    parent.appendChild(this.canvas);
  }

  /** Forget the previous track's speeds and crash dots. */
  reset(): void {
    this.speeds.clear();
    this.crashes.length = 0;
  }

  note(track: TrackSpec, s: number, speed: number, crash: boolean, x: number, z: number): void {
    const bin = Math.round(s);
    const prev = this.speeds.get(bin) ?? speed;
    this.speeds.set(bin, prev * 0.8 + speed * 0.2);
    if (crash) this.crashes.push({ x, z });
    if (this.crashes.length > 80) this.crashes.shift();
    this.draw(track);
  }

  draw(track: TrackSpec): void {
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
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
    const sx = (w - 16) / Math.max(1, maxX - minX);
    const sz = (h - 16) / Math.max(1, maxZ - minZ);
    const sc = Math.min(sx, sz);
    const ox = 8 - minX * sc;
    const oz = 8 - minZ * sc;
    ctx.lineWidth = 3;
    for (let i = 1; i < track.samples.length; i++) {
      const a = track.samples[i - 1]!;
      const b = track.samples[i]!;
      const v = this.speeds.get(Math.round(b.s)) ?? 0;
      const t = Math.max(0, Math.min(1, v / 6));
      ctx.strokeStyle = `rgb(${Math.round(40 + t * 215)}, ${Math.round(180 - t * 80)}, ${Math.round(80 + (1 - t) * 80)})`;
      ctx.beginPath();
      ctx.moveTo(ox + a.x * sc, oz + a.z * sc);
      ctx.lineTo(ox + b.x * sc, oz + b.z * sc);
      ctx.stroke();
    }
    ctx.fillStyle = '#ff5d73';
    for (const c of this.crashes) {
      ctx.fillRect(ox + c.x * sc - 1.5, oz + c.z * sc - 1.5, 3, 3);
    }
  }
}
