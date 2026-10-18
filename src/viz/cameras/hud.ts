export interface HudInput {
  speed: number;
  lap: number;
  lapTime: number;
  bestLap: number | null;
  steer: number;
  throttle: number;
  slipRear: number;
}

export interface HudReadout {
  speed: string;
  lap: string;
  lapTime: string;
  delta: string;
  steer: number;
  throttle: number;
}

export function hudReadout(h: HudInput): HudReadout {
  const best = h.bestLap;
  const delta = best == null ? '—' : `${h.lapTime - best >= 0 ? '+' : ''}${(h.lapTime - best).toFixed(2)}`;
  return {
    speed: (h.speed * 3.6).toFixed(1),
    lap: String(Math.max(1, h.lap + 1)),
    lapTime: h.lapTime.toFixed(2),
    delta,
    steer: Math.max(-1, Math.min(1, h.steer)),
    throttle: Math.max(-1, Math.min(1, h.throttle)),
  };
}

/** DOM HUD. Values come from hudReadout so tests can match CarState without a canvas. */
export class Hud {
  readonly el: HTMLElement;
  private speed: HTMLElement;
  private lap: HTMLElement;
  private time: HTMLElement;
  private delta: HTMLElement;
  private steer: HTMLElement;
  private throttle: HTMLElement;
  private map: HTMLCanvasElement;

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'hud';
    this.el.innerHTML = `
      <div class="hud-speed"><b data-hud="speed">0.0</b><span>km/h</span></div>
      <div class="hud-row"><span data-hud="lap">lap 1</span><span data-hud="time">0.00</span><span data-hud="delta">—</span></div>
      <div class="hud-bars"><i data-hud="steer"></i><i data-hud="throttle"></i></div>
      <canvas data-hud="map" width="120" height="80"></canvas>
    `;
    parent.appendChild(this.el);
    this.speed = this.el.querySelector('[data-hud="speed"]')!;
    this.lap = this.el.querySelector('[data-hud="lap"]')!;
    this.time = this.el.querySelector('[data-hud="time"]')!;
    this.delta = this.el.querySelector('[data-hud="delta"]')!;
    this.steer = this.el.querySelector('[data-hud="steer"]')!;
    this.throttle = this.el.querySelector('[data-hud="throttle"]')!;
    this.map = this.el.querySelector('[data-hud="map"]')!;
  }

  update(h: HudInput, mapPts?: { x: number; z: number }[], car?: { x: number; z: number }): void {
    const r = hudReadout(h);
    this.speed.textContent = r.speed;
    this.lap.textContent = `lap ${r.lap}`;
    this.time.textContent = r.lapTime;
    this.delta.textContent = r.delta;
    this.steer.style.transform = `translateX(${r.steer * 40}%)`;
    this.throttle.style.transform = `scaleX(${Math.abs(r.throttle)})`;
    if (mapPts && car) this.drawMap(mapPts, car);
  }

  private drawMap(pts: { x: number; z: number }[], car: { x: number; z: number }): void {
    const ctx = this.map.getContext('2d');
    if (!ctx || pts.length < 2) return;
    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (const p of pts) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minZ = Math.min(minZ, p.z);
      maxZ = Math.max(maxZ, p.z);
    }
    const sx = 100 / Math.max(1, maxX - minX);
    const sz = 64 / Math.max(1, maxZ - minZ);
    const s = Math.min(sx, sz);
    const ox = 10 - minX * s;
    const oz = 8 - minZ * s;
    ctx.clearRect(0, 0, 120, 80);
    ctx.strokeStyle = '#efeae2';
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = ox + p.x * s;
      const y = oz + p.z * s;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = '#ffb020';
    ctx.beginPath();
    ctx.arc(ox + car.x * s, oz + car.z * s, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}
