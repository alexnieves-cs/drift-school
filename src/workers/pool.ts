/** Opens the learner and env workers. The showcase path trains inside the learner worker. */

export function openLearner(): Worker {
  return new Worker(new URL('./learner.worker.ts', import.meta.url), { type: 'module' });
}

export function openEnv(): Worker {
  return new Worker(new URL('./env.worker.ts', import.meta.url), { type: 'module' });
}

export function ping(worker: Worker, t = 1): Promise<number> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('worker ping timeout')), 8000);
    const onMessage = (ev: MessageEvent) => {
      if (ev.data?.type === 'pong') {
        clearTimeout(timer);
        worker.removeEventListener('message', onMessage);
        resolve(Number(ev.data.t));
      }
    };
    worker.addEventListener('message', onMessage);
    worker.postMessage({ type: 'ping', t });
  });
}
