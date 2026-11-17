/// <reference lib="webworker" />
/**
 * Runs the coordinator's learning (GDA, engine/gda.ts) off the main thread and posts a snapshot of the plan and the
 * prices every `chunk` iterations, so that the page can animate the plan as it is learned.
 */
import { GdaState, DEFAULT_HYPER } from '../engine/gda';
import { buildModel, type Params } from '../engine/model';

type Msg =
  | { type: 'init'; params: Params; maxIter: number; chunk: number; run: number }
  | { type: 'run' }
  | { type: 'pause' }
  | { type: 'chunk'; chunk: number };

let state: GdaState | null = null;
let running = false;
let maxIter = 1_200_000;
let chunk = 20_000;
let token = 0;
let runId = 0;

function post() {
  if (!state) return;
  (self as unknown as Worker).postMessage({ type: 'snapshot', run: runId, iter: state.iter, y: state.plan(), lam: state.prices(), done: state.iter >= maxIter });
}

function loop(my: number) {
  if (!running || !state || my !== token) return;
  const n = Math.min(chunk, maxIter - state.iter);
  if (n <= 0) {
    running = false;
    post();
    return;
  }
  state.run(n);
  post();
  setTimeout(() => loop(my), 0);
}

self.onmessage = (e: MessageEvent<Msg>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    token++;
    runId = msg.run;
    state = new GdaState(buildModel(msg.params), DEFAULT_HYPER);
    maxIter = msg.maxIter;
    chunk = msg.chunk;
    running = false;
    post();
  } else if (msg.type === 'run') {
    if (!running) {
      running = true;
      loop(token);
    }
  } else if (msg.type === 'pause') {
    running = false;
  } else if (msg.type === 'chunk') {
    chunk = msg.chunk;
  }
};
