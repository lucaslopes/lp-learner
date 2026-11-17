/**
 * Occupation measures and the policies they induce (lp_learner.core.policy).
 * An occupation measure y[t][s][u] is the fraction of booths in state s taking action u at period t; it induces the
 * Markov policy pi_t(u | s) = y[t][s][u] / m_s(t). States without mass inherit the policy of the nearest state with
 * mass at the same period (ties toward the smaller state); a period without mass uses (a, b) = (0, 0).
 */
import { U } from './model';

export type Occupation = number[][][]; // [t][s][u]

export function stageMass(y: Occupation): number[][] {
  return y.map((stage) => stage.map((row) => row.reduce((acc, v) => acc + Math.max(v, 0), 0)));
}

export function conditionalPolicy(y: Occupation, eps = 1e-9): number[][][] {
  const T = y.length;
  const S = y[0].length;
  const mass = stageMass(y);
  const pi: number[][][] = Array.from({ length: T }, () => Array.from({ length: S }, () => new Array(U).fill(0)));
  for (let t = 0; t < T; t++) {
    const support: number[] = [];
    for (let s = 0; s < S; s++) if (mass[t][s] > eps) support.push(s);
    if (support.length === 0) {
      for (let s = 0; s < S; s++) pi[t][s][0] = 1;
      continue;
    }
    for (let s = 0; s < S; s++) {
      let src = s;
      if (!(mass[t][s] > eps)) {
        let best = Infinity;
        for (const k of support) {
          const score = Math.abs(k - s) + 1e-3 * (k > s ? 1 : 0);
          if (score < best) {
            best = score;
            src = k;
          }
        }
      }
      for (let u = 0; u < U; u++) pi[t][s][u] = Math.max(y[t][src][u], 0) / mass[t][src];
    }
  }
  return pi;
}

/** Normalise each period of an occupation measure to total mass one (as the pie plots do). */
export function normalisePerPeriod(y: Occupation): Occupation {
  return y.map((stage) => {
    const total = stage.reduce((acc, row) => acc + row.reduce((a, v) => a + Math.max(v, 0), 0), 0);
    return stage.map((row) => row.map((v) => (total > 0 ? Math.max(v, 0) / total : 0)));
  });
}
