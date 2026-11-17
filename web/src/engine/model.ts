/**
 * The toll-plaza model: one booth's MDP (concurrent service, "CJS"), in the conventions of lp_learner.core.
 *
 * - states s = 0..K (cars in the lane), S = K + 1;
 * - actions u = 2a + b: a = open the gate (admit one car of the platoon), b = call a mobile attendant;
 * - P[u][s][s'] transition probabilities between two platoons, c[s][u] one-period cost;
 * - resources: 0 = cars of the platoon (budget alpha per booth), 1 = attendants (budget beta per booth).
 *
 * The period length tau is Geometric(p) time steps; in each step each car leaves with probability q(b).
 * The kernel mixes Binomial(s+, (1-q)^tau) over tau, truncated when the geometric tail is below 1e-15 and renormalised,
 * exactly as lp_learner.core.instance.transition_kernel.
 */

export interface Params {
  K: number;
  T: number;
  p: number;
  bLow: number;
  bHigh: number;
  alpha: number;
  beta: number;
  gamma: number;
  gammaBar: number;
}

export const ADMIT = [0, 0, 1, 1] as const;
export const HIGH = [0, 1, 0, 1] as const;
export const U = 4;

export type Kernel = number[][][]; // [a][s][s']

export interface Model {
  params: Params;
  S: number;
  T: number;
  P: number[][][]; // [u][s][s']
  c: number[][]; // [s][u]
  budgets: [number, number];
  m0: number[];
}

export function serviceProbability(b: number, params: Params): number {
  return params.bLow + b * (params.bHigh - params.bLow);
}

function choose(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}

/** Binomial(n, x) probability of k, with 0^0 = 1. */
export function binomPmf(n: number, k: number, x: number): number {
  if (k < 0 || k > n) return 0;
  const a = k === 0 ? 1 : Math.pow(x, k);
  const b = n - k === 0 ? 1 : Math.pow(1 - x, n - k);
  return choose(n, k) * a * b;
}

const kernelCache = new Map<string, Kernel>();

/** Embedded transition kernel [a][s][s'] for a fixed per-step service probability q. */
export function transitionKernel(K: number, p: number, q: number, tol = 1e-15): Kernel {
  const key = `${K}|${p}|${q}|${tol}`;
  const hit = kernelCache.get(key);
  if (hit) return hit;
  const S = K + 1;
  let taus: number[];
  let weights: number[];
  if (p >= 1) {
    taus = [1];
    weights = [1];
  } else {
    const tauMax = Math.ceil(Math.log(tol) / Math.log(1 - p)) + 1;
    const n = Math.max(tauMax, 2);
    taus = Array.from({ length: n }, (_, i) => i + 1);
    weights = taus.map((t) => p * Math.pow(1 - p, t - 1));
  }
  const kernel: Kernel = [0, 1].map(() => Array.from({ length: S }, () => new Array(S).fill(0)));
  for (let a = 0; a < 2; a++) {
    for (let s = 0; s < S; s++) {
      const n = s < K ? s + a : s; // an admission to a full lane is blocked
      const row = kernel[a][s];
      for (let i = 0; i < taus.length; i++) {
        const remain = Math.pow(1 - q, taus[i]);
        for (let left = 0; left <= n; left++) row[left] += weights[i] * binomPmf(n, left, remain);
      }
      const total = row.reduce((x, y) => x + y, 0);
      for (let j = 0; j < S; j++) row[j] /= total;
    }
  }
  kernelCache.set(key, kernel);
  return kernel;
}

/** Probability that a given car leaves within one period: 1 - E[(1-q)^tau] = q / (1 - (1-p)(1-q)). */
export function perPeriodDeparture(p: number, q: number): number {
  return q / (1 - (1 - p) * (1 - q));
}

export function oneStageCost(params: Params, s: number, a: number, b: number): number {
  const q = serviceProbability(b, params);
  let cost = s / params.K + 2 * (1 + q) + params.gamma * (a === 0 ? 1 : 0);
  if (s === params.K && a === 1) cost += params.gammaBar;
  return cost;
}

export function costParts(params: Params, s: number, a: number, b: number) {
  const q = serviceProbability(b, params);
  const waiting = s / params.K;
  const staff = 2 * (1 + q);
  const missed = a === 0 ? params.gamma : s === params.K ? params.gammaBar : 0;
  return { waiting, staff, missed, total: waiting + staff + missed };
}

export function buildModel(params: Params): Model {
  const S = params.K + 1;
  const P: number[][][] = [];
  const c: number[][] = Array.from({ length: S }, () => new Array(U).fill(0));
  for (let u = 0; u < U; u++) {
    const a = ADMIT[u];
    const b = HIGH[u];
    const kernel = transitionKernel(params.K, params.p, serviceProbability(b, params));
    P.push(kernel[a].map((row) => row.slice()));
    for (let s = 0; s < S; s++) c[s][u] = oneStageCost(params, s, a, b);
  }
  const m0 = new Array(S).fill(0);
  m0[0] = 1;
  return { params, S, T: params.T, P, c, budgets: [params.alpha, params.beta], m0 };
}

export function withBudgets(params: Params, alpha: number, beta: number): Params {
  return { ...params, alpha, beta };
}

/** Integer capacities floor(alpha N) and floor(beta N), with the same tolerance as the Python code. */
export function capacities(N: number, alpha: number, beta: number): [number, number] {
  return [Math.floor(alpha * N + 1e-9), Math.floor(beta * N + 1e-9)];
}
