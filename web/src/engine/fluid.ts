/**
 * The large-N ("fluid") evaluation of a requesting policy under a rule for excess requests
 * (lp_learner.commons.mechanisms.fluid_mechanism). As N grows, the fraction of booths in each state evolves
 * deterministically; requests above a budget are granted with probability budget / requests (rationing) or, for the
 * attendants under congestion, every caller gets the shared speed q_eff = q(0) + (q(1) - q(0)) beta / requests.
 */
import { ADMIT, HIGH, U, transitionKernel, type Model } from './model';

export type Mechanism = 'M0' | 'M1' | 'M2';

export interface FluidResult {
  cost: number;
  components: { storage: number; processing: number; rejection: number; blocked: number };
  mass: number[][];
  usage: number[][]; // [resource][t] realised
  requested: number[][];
  grantA: number[];
  grantB: number[];
  qHigh: number[];
  lostPerPeriod: number[]; // alpha minus realised admissions to lanes that are not full
}

/** Distribution of the realised action for each state, given requests piT[s][u] and the grant probabilities. */
export function realisedPolicy(piT: number[][], grantA: number, grantB: number, mech: Mechanism): number[][] {
  return piT.map((row) => {
    const out = new Array(U).fill(0);
    for (let u = 0; u < U; u++) {
      const aReq = ADMIT[u];
      const bReq = HIGH[u];
      const aOpts: [number, number][] = aReq ? [[1, grantA], [0, 1 - grantA]] : [[0, 1]];
      const bOpts: [number, number][] = bReq && mech === 'M1' ? [[1, grantB], [0, 1 - grantB]] : [[bReq, 1]];
      for (const [ar, wa] of aOpts) for (const [br, wb] of bOpts) if (wa * wb > 0) out[2 * ar + br] += row[u] * wa * wb;
    }
    return out;
  });
}

export function fluidEvaluate(model: Model, piReq: number[][][], mech: Mechanism): FluidResult {
  const { params } = model;
  const T = piReq.length;
  const S = model.S;
  const K = params.K;
  const [alpha, beta] = model.budgets;
  const proc = [0, 1, 2, 3].map((u) => 2 * (1 + (HIGH[u] ? params.bHigh : params.bLow)));
  let m = model.m0.slice();
  const res: FluidResult = {
    cost: 0,
    components: { storage: 0, processing: 0, rejection: 0, blocked: 0 },
    mass: [],
    usage: [new Array(T).fill(0), new Array(T).fill(0)],
    requested: [new Array(T).fill(0), new Array(T).fill(0)],
    grantA: new Array(T).fill(1),
    grantB: new Array(T).fill(1),
    qHigh: new Array(T).fill(params.bHigh),
    lostPerPeriod: new Array(T).fill(0),
  };
  for (let t = 0; t < T; t++) {
    res.mass.push(m.slice());
    let reqA = 0;
    let reqB = 0;
    for (let s = 0; s < S; s++)
      for (let u = 0; u < U; u++) {
        reqA += m[s] * piReq[t][s][u] * ADMIT[u];
        reqB += m[s] * piReq[t][s][u] * HIGH[u];
      }
    res.requested[0][t] = reqA;
    res.requested[1][t] = reqB;
    if ((mech === 'M1' || mech === 'M2') && reqA > alpha) res.grantA[t] = alpha / reqA;
    if (mech === 'M1' && reqB > beta) res.grantB[t] = beta / reqB;
    if (mech === 'M2' && reqB > beta) res.qHigh[t] = params.bLow + ((params.bHigh - params.bLow) * beta) / reqB;
    const pir = realisedPolicy(piReq[t], res.grantA[t], res.grantB[t], mech);
    let served = 0;
    for (let s = 0; s < S; s++) {
      for (let u = 0; u < U; u++) {
        const w = m[s] * pir[s][u];
        res.usage[0][t] += w * ADMIT[u];
        res.usage[1][t] += w * HIGH[u];
        res.components.processing += w * proc[u];
        res.components.rejection += w * (ADMIT[u] === 0 ? params.gamma : 0);
        if (s === K && ADMIT[u] === 1) res.components.blocked += w * params.gammaBar;
        if (s < K) served += w * ADMIT[u];
      }
      res.components.storage += (m[s] * s) / K;
    }
    res.lostPerPeriod[t] = alpha - served;
    if (t + 1 < T) {
      let P = model.P;
      if (mech === 'M2' && res.qHigh[t] < params.bHigh) {
        const congested = transitionKernel(K, params.p, res.qHigh[t]);
        P = model.P.map((rows, u) => (HIGH[u] === 1 ? congested[ADMIT[u]] : rows));
      }
      const next = new Array(S).fill(0);
      for (let s = 0; s < S; s++)
        for (let u = 0; u < U; u++) {
          const w = m[s] * pir[s][u];
          if (w === 0) continue;
          for (let s2 = 0; s2 < S; s2++) next[s2] += w * P[u][s][s2];
        }
      m = next;
    }
  }
  const c = res.components;
  res.cost = c.storage + c.processing + c.rejection + c.blocked;
  return res;
}

/**
 * Large-N occupation measure of a requesting policy: the fraction of booths in state s at period t that ask for
 * (`realised` false) or end up with (`realised` true, after the lotteries) action u.
 */
export function occupation(fl: FluidResult, piReq: number[][][], mech: Mechanism, realised: boolean): number[][][] {
  return piReq.map((piT, t) => {
    const pol = realised ? realisedPolicy(piT, fl.grantA[t], fl.grantB[t], mech) : piT;
    return pol.map((row, s) => row.map((v) => v * fl.mass[t][s]));
  });
}
