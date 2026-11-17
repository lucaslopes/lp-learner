import { useCallback, useEffect, useRef, useState } from 'react';
import type { Params } from '../engine/model';

export interface GdaSnapshot {
  iter: number;
  y: number[][][];
  lam: number[][];
  done: boolean;
}

/**
 * The coordinator learning its plan by gradient descent-ascent, in a web worker. Returns the latest snapshot and
 * controls. A new worker run starts whenever the budgets change or `reset` is called.
 */
export function useGda(params: Params, { maxIter = 1_200_000, chunk = 20_000, autoStart = false } = {}) {
  const workerRef = useRef<Worker | null>(null);
  const [snap, setSnap] = useState<GdaSnapshot | null>(null);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<GdaSnapshot[]>([]);
  const runRef = useRef(0);
  const key = `${params.alpha}|${params.beta}`;

  useEffect(() => {
    const worker = new Worker(new URL('../workers/gda.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.onmessage = (e: MessageEvent<GdaSnapshot & { type: string; run: number }>) => {
      if (e.data.type !== 'snapshot' || e.data.run !== runRef.current) return; // a snapshot of an earlier run
      const s = { iter: e.data.iter, y: e.data.y, lam: e.data.lam, done: e.data.done };
      setSnap(s);
      setHistory((h) => (h.length && h[h.length - 1].iter === s.iter ? h : [...h, s]));
      if (s.done) setRunning(false);
    };
    runRef.current += 1;
    setSnap(null);
    setHistory([]);
    worker.postMessage({ type: 'init', params, maxIter, chunk, run: runRef.current });
    if (autoStart) {
      worker.postMessage({ type: 'run' });
      setRunning(true);
    } else setRunning(false);
    return () => worker.terminate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, maxIter]);

  const run = useCallback(() => {
    workerRef.current?.postMessage({ type: 'run' });
    setRunning(true);
  }, []);
  const pause = useCallback(() => {
    workerRef.current?.postMessage({ type: 'pause' });
    setRunning(false);
  }, []);
  const reset = useCallback(() => {
    runRef.current += 1;
    setSnap(null);
    setHistory([]);
    workerRef.current?.postMessage({ type: 'init', params, maxIter, chunk, run: runRef.current });
    setRunning(false);
  }, [params, maxIter, chunk]);
  const setChunk = useCallback((c: number) => workerRef.current?.postMessage({ type: 'chunk', chunk: c }), []);

  return { snap, history, running, run, pause, reset, setChunk, maxIter };
}
