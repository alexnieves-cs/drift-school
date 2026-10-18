import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

/**
 * Two looks for the same scene.
 *
 *  - lit (High): Lambert shading with sun and sky, colors in sRGB hex that three converts to its
 *    linear working space, AgX tone mapping and sRGB output.
 *  - unlit (Low): flat colors with tone mapping off and linear output, so a hex color reaches the
 *    screen as written. For flat unlit surfaces that is the same pixel as the managed pipeline would
 *    give, minus lighting and AgX. It matters because those per-pixel transcendentals dominate fill
 *    cost on weak or software rasterizers.
 */

export const SKY_HEX = 0x10161c;
export const GROUND_LIT_HEX = 0x163528;
export const GROUND_UNLIT_HEX = 0x17382b;
export const PALETTE = [0x5b8def, 0x3dd68c, 0xff5d73, 0xc084fc, 0x22d3ee, 0xf97316, 0xeab308, 0x94a3b8];

/** A hex color in the space the shaders of this look expect. */
export function paint(hex: number, unlit: boolean): THREE.Color {
  return unlit ? new THREE.Color().setHex(hex, THREE.LinearSRGBColorSpace) : new THREE.Color(hex);
}

export interface LookMaterials {
  ground: THREE.Material;
  ghost: THREE.Material;
  hero: THREE.Material;
  replay: THREE.Material;
  trail: THREE.Material;
  marks: THREE.Material;
}

export function makeLook(unlit: boolean): LookMaterials {
  const translucent = { transparent: true, depthWrite: false };
  if (unlit) {
    return {
      ground: new THREE.MeshBasicMaterial({ color: paint(GROUND_UNLIT_HEX, true), fog: false }),
      ghost: new THREE.MeshBasicMaterial({ ...translucent, opacity: 0.55 }),
      hero: new THREE.MeshBasicMaterial({ vertexColors: true }),
      replay: new THREE.MeshBasicMaterial({ ...translucent, color: paint(0x9fe8ff, true), opacity: 0.85 }),
      trail: new THREE.LineBasicMaterial({ color: paint(0xffb020, true) }),
      marks: new THREE.MeshBasicMaterial({ color: paint(0x141414, true) }),
    };
  }
  return {
    ground: new THREE.MeshLambertMaterial({ color: GROUND_LIT_HEX }),
    ghost: new THREE.MeshLambertMaterial({ ...translucent, opacity: 0.45 }),
    hero: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.4, metalness: 0.3 }),
    replay: new THREE.MeshLambertMaterial({ ...translucent, color: 0x9fe8ff, opacity: 0.85 }),
    trail: new THREE.LineBasicMaterial({ color: 0xffb020 }),
    marks: new THREE.MeshBasicMaterial({ color: 0x141414 }),
  };
}

/** Merged hero/replay car: body, cabin and four wheels as one draw call, vertex colored. */
export function carGeometry(unlit: boolean): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const paintPart = (g: THREE.BufferGeometry, hex: number) => {
    const c = paint(hex, unlit);
    const n = g.getAttribute('position').count;
    const arr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      arr[i * 3] = c.r;
      arr[i * 3 + 1] = c.g;
      arr[i * 3 + 2] = c.b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(arr, 3));
    return g;
  };
  const body = new THREE.BoxGeometry(0.36, 0.06, 0.2);
  body.translate(0, 0.05, 0);
  parts.push(paintPart(body, 0xffb020));
  const cabin = new THREE.BoxGeometry(0.14, 0.05, 0.14);
  cabin.translate(-0.02, 0.1, 0);
  parts.push(paintPart(cabin, 0x1a1e27));
  for (const [x, z] of [[0.1, 0.09], [0.1, -0.09], [-0.1, 0.09], [-0.1, -0.09]] as const) {
    const wheel = new THREE.CylinderGeometry(0.035, 0.035, 0.03, 10);
    wheel.rotateX(Math.PI / 2);
    wheel.translate(x, 0.035, z);
    parts.push(paintPart(wheel, 0x111111));
  }
  const merged = mergeParts(parts);
  parts.forEach((p) => p.dispose());
  return merged;
}


function mergeParts(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  return mergeGeometries(parts.map((p) => p.toNonIndexed()), false) ?? new THREE.BoxGeometry(0.36, 0.06, 0.2);
}
