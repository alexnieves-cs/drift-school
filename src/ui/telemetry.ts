export interface TelemetrySample {
  speed: number;
  steer: number;
  throttle: number;
  slip: number[];
  load: number[];
  ax: number;
  ay: number;
}

export class TelemetryPanel {
  private bars: HTMLElement;
  private gg: HTMLCanvasElement;
  private pts: { x: number; y: number }[] = [];

  constructor(parent: HTMLElement) {
    const root = document.createElement('div');
    root.className = 'telemetry';
    root.innerHTML = `<div class="tel-bars"></div><canvas class="gg" width="140" height="140"></canvas>`;
    parent.appendChild(root);
    this.bars = root.querySelector('.tel-bars')!;
    this.gg = root.querySelector('canvas')!;
  }

  update(s: TelemetrySample): void {
    const labels = ['FL', 'FR', 'RL', 'RR'];
    this.bars.innerHTML = labels
      .map((name, i) => {
        const load = Math.min(1, (s.load[i] ?? 0) / 8);
        const slip = s.slip[i] ?? 0;
        return `<div><b>${name}</b><span style="width:${load * 100}%"></span><em>${((slip * 180) / Math.PI).toFixed(0)}°</em></div>`;
      })
      .join('');
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
