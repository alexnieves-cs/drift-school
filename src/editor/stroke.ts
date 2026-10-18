export interface V2 {
  x: number;
  z: number;
}

function dist(a: V2, b: V2): number {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

/** Ramer–Douglas–Peucker. */
export function rdp(points: V2[], epsilon: number): V2[] {
  if (points.length < 3) return points.slice();
  let maxD = 0;
  let idx = 0;
  const a = points[0]!;
  const b = points[points.length - 1]!;
  const abx = b.x - a.x;
  const abz = b.z - a.z;
  const ab = abx * abx + abz * abz || 1;
  for (let i = 1; i < points.length - 1; i++) {
    const p = points[i]!;
    const t = Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.z - a.z) * abz) / ab));
    const d = Math.hypot(p.x - (a.x + abx * t), p.z - (a.z + abz * t));
    if (d > maxD) {
      maxD = d;
      idx = i;
    }
  }
  if (maxD > epsilon) {
    const left = rdp(points.slice(0, idx + 1), epsilon);
    const right = rdp(points.slice(idx), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [a, b];
}

/** Chaikin corner cutting. One pass. */
export function chaikin(points: V2[], closed = false): V2[] {
  const out: V2[] = [];
  const n = points.length;
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    const a = points[i]!;
    const b = points[(i + 1) % n]!;
    out.push({ x: a.x * 0.75 + b.x * 0.25, z: a.z * 0.75 + b.z * 0.25 });
    out.push({ x: a.x * 0.25 + b.x * 0.75, z: a.z * 0.25 + b.z * 0.75 });
  }
  if (!closed) {
    out.unshift(points[0]!);
    out.push(points[n - 1]!);
  }
  return out;
}

export function smoothStroke(points: V2[], closed: boolean): V2[] {
  const simple = rdp(points, 0.25);
  let cur = simple;
  for (let i = 0; i < 2; i++) cur = chaikin(cur, closed);
  return cur;
}

export function toFloat(points: V2[]): Float32Array {
  const out = new Float32Array(points.length * 2);
  points.forEach((p, i) => {
    out[i * 2] = p.x;
    out[i * 2 + 1] = p.z;
  });
  return out;
}

export function fromFloat(data: ArrayLike<number>): V2[] {
  const out: V2[] = [];
  for (let i = 0; i + 1 < data.length; i += 2) out.push({ x: data[i]!, z: data[i + 1]! });
  return out;
}

export { dist };
