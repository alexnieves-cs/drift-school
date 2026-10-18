export interface Checkpoint {
  name: string;
  gen: number;
  stage: number;
  weights: number[][];
}

const memory = new Map<string, Checkpoint>();
const DB = 'drift-school';
const STORE = 'checkpoints';

function clone(cp: Checkpoint): Checkpoint {
  return { name: cp.name, gen: cp.gen, stage: cp.stage, weights: cp.weights.map((w) => w.slice()) };
}

function idb(): Promise<IDBDatabase> | null {
  if (typeof indexedDB === 'undefined') return null;
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'name' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveCheckpoint(cp: Checkpoint): Promise<void> {
  const copy = clone(cp);
  memory.set(copy.name, copy);
  const dbp = idb();
  if (!dbp) return;
  const db = await dbp;
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(copy);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function loadCheckpoint(name: string): Promise<Checkpoint | null> {
  if (memory.has(name)) return clone(memory.get(name)!);
  const dbp = idb();
  if (!dbp) return null;
  const db = await dbp;
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(name);
    req.onsuccess = () => {
      const v = req.result as Checkpoint | undefined;
      if (v) memory.set(name, clone(v));
      resolve(v ? clone(v) : null);
    };
    req.onerror = () => reject(req.error);
  });
}

/** Fork preserves the base checkpoint and writes a new name. */
export async function forkCheckpoint(from: string, to: string): Promise<Checkpoint | null> {
  const src = await loadCheckpoint(from);
  if (!src) return null;
  const next = clone(src);
  next.name = to;
  await saveCheckpoint(next);
  return next;
}

export function memoryCheckpoint(name: string): Checkpoint | null {
  const c = memory.get(name);
  return c ? clone(c) : null;
}
