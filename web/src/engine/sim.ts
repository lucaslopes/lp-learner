/**
 * A plaza of N booths, simulated one period at a time (as lp_learner.commons.experiment does, but recording what the
 * animation needs).
 *
 * Each period: every booth's controller asks for a car (a) and an attendant (b); a rule resolves requests above the
 * budgets (rationing: a lottery among the askers; congestion: cars by lottery, attendants shared by all callers at the
 * speed q_eff); a car enters each lane whose gate is open and that is not full; then each lane is served for its own
 * random number tau ~ Geometric(p) of time steps, and in every step each car leaves with probability q(b). Drawing tau
 * and the departure step of each car gives exactly the transition probabilities P of the model (independent clocks per
 * lane, as in the tutorial). Costs are the model's one-period costs of the realised actions.
 */
import { ADMIT, HIGH, U, capacities, serviceProbability, type Model } from './model';
import { Rng } from './rng';

export type Rule = 'rationing' | 'congestion';

export interface Actions {
  a: number[];
  b: number[];
}

export interface Controller {
  name: string;
  act(t: number, states: number[], rng: Rng): Actions;
}

/** Every booth follows the same deterministic table u[t][s] (selfish or price-taking booths). */
export function tableController(name: string, u: number[][]): Controller {
  return {
    name,
    act(t, states) {
      return { a: states.map((s) => ADMIT[u[t][s]]), b: states.map((s) => HIGH[u[t][s]]) };
    },
  };
}

/** Every booth flips its own coins with the plan's probabilities (budgets hold only on average). */
export function independentController(pi: number[][][]): Controller {
  return {
    name: 'independent',
    act(t, states, rng) {
      const a: number[] = [];
      const b: number[] = [];
      for (const s of states) {
        const x = rng.next();
        let acc = 0;
        let pick = U - 1;
        for (let u = 0; u < U; u++) {
          acc += pi[t][s][u];
          if (x < acc) {
            pick = u;
            break;
          }
        }
        a.push(ADMIT[pick]);
        b.push(HIGH[pick]);
      }
      return { a, b };
    },
  };
}

/** Systematic randomised rounding of each row z[s][:] to integers with the same (integer) row sum and E[n] = z. */
export function systematicRound(z: number[][], rng: Rng): number[][] {
  return z.map((row) => {
    const base = row.map((v) => Math.floor(v + 1e-12));
    const frac = row.map((v, i) => Math.max(v - base[i], 0));
    const total = Math.round(row.reduce((x, y) => x + y, 0));
    const roundUps = total - base.reduce((x, y) => x + y, 0);
    const counts = base.slice();
    if (roundUps > 0) {
      const shift = rng.next();
      const cum: number[] = [];
      let acc = 0;
      for (const f of frac) cum.push((acc += f));
      for (let k = 0; k < roundUps; k++) {
        const point = shift + k;
        let idx = cum.findIndex((c) => c > point);
        if (idx < 0) idx = row.length - 1;
        counts[idx]++;
      }
    }
    return counts;
  });
}

export interface RoundingTrace {
  counts: number[]; // booths per state
  targets: number[][]; // N_s pi
  rounded: number[][];
  repaired: number[][];
  movesCars: number;
  movesAtt: number;
}

/** Greedy repair of integer counts n[s][u] onto the budgets (lp_learner.finite_n.policies.repair_counts). */
export function repairCounts(n: number[][], piT: number[][], capA: number, capB: number): { n: number[][]; movesA: number; movesB: number } {
  const out = n.map((r) => r.slice());
  const S = out.length;
  const pa = piT.map((r) => r[2] + r[3]);
  const pb = piT.map((r) => r[1] + r[3]);
  let movesA = 0;
  let movesB = 0;
  const totalA = () => out.reduce((acc, r) => acc + r[2] + r[3], 0);
  const totalB = () => out.reduce((acc, r) => acc + r[1] + r[3], 0);
  while (totalA() > capA) {
    let best = -1;
    for (let s = 0; s < S; s++) {
      if (out[s][2] + out[s][3] <= 0) continue;
      if (best < 0 || pa[s] < pa[best] || (pa[s] === pa[best] && s > best)) best = s; // smallest pa, ties: largest s
    }
    const u = [2, 3].filter((k) => out[best][k] > 0).sort((x, y) => piT[best][x] - piT[best][y] || x - y)[0];
    out[best][u]--;
    out[best][u - 2]++;
    movesA++;
  }
  while (totalB() > capB) {
    let best = -1;
    for (let s = 0; s < S; s++) {
      if (out[s][1] + out[s][3] <= 0) continue;
      if (best < 0 || pb[s] < pb[best]) best = s; // smallest pb, ties: smallest s
    }
    const u = [1, 3].filter((k) => out[best][k] > 0).sort((x, y) => piT[best][x] - piT[best][y] || x - y)[0];
    out[best][u]--;
    out[best][u - 1]++;
    movesB++;
  }
  return { n: out, movesA, movesB };
}

/** The coordinator: round the plan's targets N_s pi_t(.|s), repair onto the budgets, hand out in random order. */
export function projectionController(pi: number[][][], budgets: [number, number], onTrace?: (trace: RoundingTrace) => void): Controller {
  return {
    name: 'coordinator',
    act(t, states, rng) {
      const N = states.length;
      const S = pi[0].length;
      const [capA, capB] = capacities(N, budgets[0], budgets[1]);
      const counts = new Array(S).fill(0);
      for (const s of states) counts[s]++;
      const targets = counts.map((ns, s) => pi[t][s].map((p) => ns * p));
      const rounded = systematicRound(targets, rng);
      const { n, movesA, movesB } = repairCounts(rounded, pi[t], capA, capB);
      onTrace?.({ counts, targets, rounded, repaired: n.map((r) => r.slice()), movesCars: movesA, movesAtt: movesB });
      const byState: number[][] = Array.from({ length: S }, () => []);
      states.forEach((s, i) => byState[s].push(i));
      const a = new Array(N).fill(0);
      const b = new Array(N).fill(0);
      for (let s = 0; s < S; s++) {
        const booths = rng.shuffle(byState[s].slice());
        let k = 0;
        for (let u = 0; u < U; u++)
          for (let j = 0; j < n[s][u]; j++, k++) {
            a[booths[k]] = ADMIT[u];
            b[booths[k]] = HIGH[u];
          }
      }
      return { a, b };
    },
  };
}

export interface LaneOutcome {
  s: number;
  aReq: number;
  bReq: number;
  a: number; // realised
  b: number; // realised
  entered: boolean; // a new car joined the lane
  blocked: boolean; // gate open, lane full: the car was lost
  missed: boolean; // no new car this period (missed toll)
  tau: number; // time steps of service until the next platoon
  departures: number[]; // per car in the lane after entry (front first): leaving step 1..tau, or 0 if it stays
  sNext: number;
  q: number; // per-step probability that a car leaves
  cost: number;
}

export interface PeriodOutcome {
  t: number;
  rushHour: number;
  lanes: LaneOutcome[];
  capA: number;
  capB: number;
  requestsA: number;
  requestsB: number;
  carsEntered: number;
  carsLost: number; // cars of the platoon that did not get into a lane
  qEff: number;
  meanCost: number;
}

/** Choose min(#requests, cap) requesters uniformly at random. */
function ration(requests: number[], cap: number, rng: Rng): number[] {
  const idx = requests.flatMap((r, i) => (r ? [i] : []));
  const served = new Array(requests.length).fill(0);
  if (idx.length <= cap) idx.forEach((i) => (served[i] = 1));
  else rng.shuffle(idx).slice(0, cap).forEach((i) => (served[i] = 1));
  return served;
}

export class Plaza {
  states: number[];
  t = 0;
  rushHour = 0;
  readonly rngMove: Rng;
  readonly rngLottery: Rng;
  readonly rngControl: Rng;

  constructor(
    readonly model: Model,
    readonly N: number,
    public rule: Rule,
    seed = 2026,
  ) {
    this.states = new Array(N).fill(0);
    const root = new Rng(seed);
    this.rngMove = root.fork(0);
    this.rngLottery = root.fork(1);
    this.rngControl = root.fork(2);
  }

  /** Simulate one period with `controller` and advance; after the last period a new rush hour starts with empty lanes. */
  step(controller: Controller): PeriodOutcome {
    const { model, N } = this;
    const { params } = model;
    const K = params.K;
    const [capA, capB] = capacities(N, model.budgets[0], model.budgets[1]);
    const req = controller.act(this.t, this.states, this.rngControl);
    const requestsA = req.a.reduce((x, y) => x + y, 0);
    const requestsB = req.b.reduce((x, y) => x + y, 0);
    const a = ration(req.a, capA, this.rngLottery);
    let b: number[];
    let qEff = params.bHigh;
    if (this.rule === 'rationing') b = ration(req.b, capB, this.rngLottery);
    else {
      b = req.b.slice();
      if (requestsB > capB && requestsB > 0) qEff = params.bLow + ((params.bHigh - params.bLow) * capB) / requestsB;
    }
    const lanes: LaneOutcome[] = [];
    let entered = 0;
    let costSum = 0;
    for (let i = 0; i < N; i++) {
      const s = this.states[i];
      const full = s === K;
      const enters = a[i] === 1 && !full;
      const n = s + (enters ? 1 : 0);
      const q = b[i] === 1 ? (this.rule === 'congestion' ? qEff : params.bHigh) : serviceProbability(0, params);
      const tau = this.rngMove.geometric(params.p);
      const departures: number[] = [];
      let stay = 0;
      for (let k = 0; k < n; k++) {
        const leaveStep = this.rngMove.geometric(q);
        if (leaveStep <= tau) departures.push(leaveStep);
        else {
          departures.push(0);
          stay++;
        }
      }
      const u = 2 * a[i] + b[i];
      const cost = model.c[s][u];
      costSum += cost;
      if (enters) entered++;
      lanes.push({
        s, aReq: req.a[i], bReq: req.b[i], a: a[i], b: b[i], entered: enters, blocked: a[i] === 1 && full,
        missed: !enters, tau, departures, sNext: stay, q, cost,
      });
    }
    const out: PeriodOutcome = {
      t: this.t, rushHour: this.rushHour, lanes, capA, capB, requestsA, requestsB, carsEntered: entered,
      carsLost: capA - entered, qEff, meanCost: costSum / N,
    };
    this.t++;
    if (this.t >= model.T) {
      this.t = 0;
      this.rushHour++;
      this.states = new Array(N).fill(0);
    } else this.states = lanes.map((l) => l.sNext);
    return out;
  }
}

/** Run whole rush hours without animation and return the cost per booth of each (for quick comparisons). */
export function runRushHours(model: Model, N: number, rule: Rule, controller: Controller, rushHours: number, seed: number): number[] {
  const plaza = new Plaza(model, N, rule, seed);
  const costs: number[] = [];
  let acc = 0;
  for (let k = 0; k < rushHours * model.T; k++) {
    acc += plaza.step(controller).meanCost;
    if (plaza.t === 0) {
      costs.push(acc);
      acc = 0;
    }
  }
  return costs;
}
