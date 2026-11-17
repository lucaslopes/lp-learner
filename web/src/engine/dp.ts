/**
 * Backward induction for one booth (lp_learner.commons.selfish.private_dp): minimise the expected sum of a
 * stage-dependent cost over the horizon, with zero terminal value. Exact ties (within TIE_TOL) are broken by an explicit
 * rule: 'no_request' picks the minimiser that asks for fewer resources (then no car, then no attendant); 'request' picks
 * the one that asks for more.
 */
import { ADMIT, HIGH, U, type Model } from './model';

export type TieBreak = 'no_request' | 'request';
export const TIE_TOL = 1e-6;
const REQUESTS = [0, 1, 1, 2];

export interface DpSolution {
  V: number[][]; // [T+1][S]
  Q: number[][][]; // [T][S][U]
  u: number[][]; // [T][S] chosen action index
  nMinimisers: number[][];
  tieBreak: TieBreak;
}

function greedy(q: number[], tie: TieBreak, tol: number): [number, number] {
  const best = Math.min(...q);
  const cands: number[] = [];
  for (let u = 0; u < q.length; u++) if (q[u] <= best + tol) cands.push(u);
  const key = (u: number) =>
    tie === 'no_request' ? [REQUESTS[u], ADMIT[u], HIGH[u], u] : [-REQUESTS[u], -ADMIT[u], -HIGH[u], u];
  cands.sort((x, y) => {
    const kx = key(x);
    const ky = key(y);
    for (let i = 0; i < kx.length; i++) if (kx[i] !== ky[i]) return kx[i] - ky[i];
    return 0;
  });
  return [cands[0], cands.length];
}

export function privateDp(P: number[][][], cost: number[][][], tie: TieBreak = 'no_request', tol = TIE_TOL): DpSolution {
  const T = cost.length;
  const S = cost[0].length;
  const V: number[][] = Array.from({ length: T + 1 }, () => new Array(S).fill(0));
  const Q: number[][][] = Array.from({ length: T }, () => Array.from({ length: S }, () => new Array(U).fill(0)));
  const u: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  const nMin: number[][] = Array.from({ length: T }, () => new Array(S).fill(0));
  for (let t = T - 1; t >= 0; t--) {
    for (let s = 0; s < S; s++) {
      for (let a = 0; a < U; a++) {
        let future = 0;
        for (let s2 = 0; s2 < S; s2++) future += P[a][s][s2] * V[t + 1][s2];
        Q[t][s][a] = cost[t][s][a] + future;
      }
      const [pick, n] = greedy(Q[t][s], tie, tol);
      u[t][s] = pick;
      nMin[t][s] = n;
      V[t][s] = Math.min(...Q[t][s]);
    }
  }
  return { V, Q, u, nMinimisers: nMin, tieBreak: tie };
}

export function stageCosts(model: Model): number[][][] {
  return Array.from({ length: model.T }, () => model.c.map((row) => row.slice()));
}

/** The selfish booth: its own cost, budgets ignored. */
export function selfishSolution(model: Model, tie: TieBreak = 'no_request'): DpSolution {
  return privateDp(model.P, stageCosts(model), tie);
}

/** The price-taking booth: its own cost plus the posted prices lam[0][t] per car and lam[1][t] per attendant. */
export function pricedSolution(model: Model, lam: number[][], tie: TieBreak = 'no_request'): DpSolution {
  const cost = stageCosts(model).map((stage, t) =>
    stage.map((row) => row.map((v, u) => v + lam[0][t] * ADMIT[u] + lam[1][t] * HIGH[u])),
  );
  return privateDp(model.P, cost, tie);
}

/** Private gain of an attendant, Q(s,(a,0)) - Q(s,(a,1)), with the gate decision a that the solution takes. */
export function attendantGain(sol: DpSolution): number[][] {
  return sol.Q.map((stage, t) =>
    stage.map((q, s) => {
      const a = ADMIT[sol.u[t][s]];
      return q[2 * a] - q[2 * a + 1];
    }),
  );
}

/** One-hot Markov policy pi[t][s][u] from a table of action indices. */
export function deterministicPolicy(u: number[][]): number[][][] {
  return u.map((row) => row.map((k) => [0, 1, 2, 3].map((j) => (j === k ? 1 : 0))));
}
