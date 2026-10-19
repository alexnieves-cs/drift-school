export interface ControlState {
  paused: boolean;
  baseline: boolean;
  cars: number;
  stage: number;
  grip: number;
  mass: number;
  dr: 'off' | 'low' | 'high';
  camera: 'pov' | 'chase' | 'topdown';
  quality: 'low' | 'high';
}

export function mountControls(parent: HTMLElement, onChange: (s: ControlState, reason: string) => void): {
  read: () => ControlState;
  setStatus: (text: string, kind?: string) => void;
} {
  parent.innerHTML = `
    <header class="brand"><span>drift-school</span><small data-status="booting">booting</small></header>
    <div class="row">
      <button type="button" data-act="pause">pause</button>
      <button type="button" data-act="baseline">baseline</button>
      <button type="button" data-act="editor">draw</button>
      <button type="button" data-act="train">train on this</button>
    </div>
    <label>cars <input data-field="cars" type="range" min="4" max="64" step="4" value="8" /></label>
    <label>stage <input data-field="stage" type="range" min="0" max="4" step="1" value="0" /></label>
    <label>grip <input data-field="grip" type="range" min="0.5" max="1.4" step="0.05" value="1" /></label>
    <label>mass <input data-field="mass" type="range" min="0.7" max="1.4" step="0.05" value="1" /></label>
    <label>DR
      <select data-field="dr"><option value="off">off</option><option value="low">low</option><option value="high">high</option></select>
    </label>
    <div class="row">
      <button type="button" data-cam="topdown">1 top</button>
      <button type="button" data-cam="chase">2 chase</button>
      <button type="button" data-cam="pov">3 pov</button>
    </div>
    <div class="row">
      <button type="button" data-q="low">low</button>
      <button type="button" data-q="high">high</button>
      <button type="button" data-act="save">save</button>
      <button type="button" data-act="load">load</button>
    </div>
    <p class="hint">keys 1/2/3 cameras · wasd on the drive page · ?inline=1 skips the worker</p>
  `;
  const status = parent.querySelector('[data-status]') as HTMLElement;
  const read = (): ControlState => ({
    paused: parent.querySelector<HTMLButtonElement>('[data-act="pause"]')!.dataset.on === '1',
    baseline: parent.querySelector<HTMLButtonElement>('[data-act="baseline"]')!.dataset.on === '1',
    cars: Number(parent.querySelector<HTMLInputElement>('[data-field="cars"]')!.value),
    stage: Number(parent.querySelector<HTMLInputElement>('[data-field="stage"]')!.value),
    grip: Number(parent.querySelector<HTMLInputElement>('[data-field="grip"]')!.value),
    mass: Number(parent.querySelector<HTMLInputElement>('[data-field="mass"]')!.value),
    dr: parent.querySelector<HTMLSelectElement>('[data-field="dr"]')!.value as ControlState['dr'],
    camera: (parent.dataset.camera as ControlState['camera']) || 'topdown',
    quality: (parent.dataset.quality as ControlState['quality']) || 'low',
  });
  const emit = (reason: string) => onChange(read(), reason);
  parent.addEventListener('click', (ev) => {
    const t = (ev.target as HTMLElement).closest('button');
    if (!t) return;
    if (t.dataset.act === 'pause' || t.dataset.act === 'baseline') {
      t.dataset.on = t.dataset.on === '1' ? '0' : '1';
      t.classList.toggle('on', t.dataset.on === '1');
      emit(t.dataset.act);
    } else if (t.dataset.cam) {
      parent.dataset.camera = t.dataset.cam;
      emit('camera');
    } else if (t.dataset.q) {
      parent.dataset.quality = t.dataset.q;
      emit('quality');
    } else if (t.dataset.act) emit(t.dataset.act);
  });
  parent.addEventListener('input', (ev) => {
    const field = (ev.target as HTMLElement).dataset.field;
    if (field) emit(field);
  });
  return {
    read,
    setStatus: (text, kind = 'live') => {
      status.textContent = text;
      status.dataset.status = kind;
    },
  };
}
