export interface TelemetrySample {
  speed: number;
  steer: number;
  throttle: number;
  slip: number[];
  load: number[];
  ax: number;
  ay: number;
}

const WHEELS = ['FL', 'FR', 'RL', 'RR'];

/** Selected-car panel. Elements are built once; update() only writes text and widths. */
export class TelemetryPanel {
  private gg: HTMLCanvasElement;
  private loadBars: HTMLElement[] = [];
  private slipText: HTMLElement[] = [];
  private pts: { x: number; y: number }[] = [];

  constructor(parent: HTMLElement) {
    const root = document.createElement('div');
    root.className = 'telemetry';
    root.innerHTML = `<div class="tel-bars">${WHEELS.map((n) => `<div><b>${n}</b><span></span><em>0°</em></div>`).join('')}</div><canvas class="gg" width="140" height="140"></canvas>`;
    parent.appendChild(root);
    root.querySelectorAll<HTMLElement>('.tel-bars span').forEach((s) => this.loadBars.push(s));
    root.querySelectorAll<HTMLElement>('.tel-bars em').forEach((s) => this.slipText.push(s));
    this.gg = root.querySelector('canvas')!;
  }

  update(s: TelemetrySample): void {
    for (let i = 0; i < WHEELS.length; i++) {
      const load = Math.min(1, (s.load[i] ?? 0) / 8);
      this.loadBars[i]!.style.width = `${(load * 100).toFixed(0)}%`;
      this.slipText[i]!.textContent = `${(((s.slip[i] ?? 0) * 180) / Math.PI).toFixed(0)}°`;
    }
    this.pts.push({ x: s.ay, y: s.ax });
    if (this.pts.length > 80) this.pts.shift();
    const ctx = this.gg.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, 140, 140);
    ctx.strokeStyle = '#243044';
    ctx.beginPath();
    ctx.arc(70, 70, 50, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = '#ffb020';
    ctx.beginPath();
    this.pts.forEach((p, i) => {
      const x = 70 + p.x * 28;
      const y = 70 - p.y * 28;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}
