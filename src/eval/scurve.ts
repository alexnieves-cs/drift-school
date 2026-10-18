/** The SPEC M4 "S-curve": a closed band with a wavy S in both long sides, as a mouse path in metres. */
export function sCurveLoop(step = 1): [number, number][] {
  const pts: [number, number][] = [];
  const wave = (x: number) => 1.5 * Math.sin((2 * Math.PI * x) / 12);
  const arc = step / 4;
  for (let x = -12; x <= 12; x += step) pts.push([x, -4 + wave(x)]);
  for (let a = -Math.PI / 2; a <= Math.PI / 2; a += arc) pts.push([12 + 4 * Math.cos(a), 4 * Math.sin(a)]);
  for (let x = 12; x >= -12; x -= step) pts.push([x, 4 + wave(x)]);
  for (let a = Math.PI / 2; a <= (3 * Math.PI) / 2; a += arc) pts.push([-12 + 4 * Math.cos(a), 4 * Math.sin(a)]);
  pts.push(pts[0]!);
  return pts;
}
