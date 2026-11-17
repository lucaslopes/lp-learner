/**
 * Gradient descent-ascent (GDA) on the regularised Lagrangian of the coordinator's LP (lp_learner.core.gda, method C),
 * with the same gradients, simultaneous updates, projections and settings. The plan y moves toward actions that are
 * cheap at the current prices; a price rises when its budget is overused and falls (down to zero) when it is not.
 *
 *   gy   = c + Gy y - lam_nn + sum_l lam_res[l][t] d_l + mu0 (t = 0) or muf[t-1] (t > 0) - sum_s' P muf[t]
 *   glr  = d y - budget - Gl lam_res        gln = -y - Gn lam_nn
 *   gm0  = sum_u y[0] - m0 - Gm mu0         gmf = sum_u y[t] - inflow(t-1 -> t) - Gm muf[t-1]
 *
 * Arrays are flat Float64Arrays for speed: y[(t*S + s)*U + u], lamRes[l*T + t], muf[(t-1)*S + s].
 */
import { ADMIT, HIGH, U, type Model } from './model';

export interface GdaHyper {
  GammaY: number;
  GammaLambda: number;
  GammaMu: number;
  etaY: number;
  etaDual: number;
}

export const DEFAULT_HYPER: GdaHyper = { GammaY: 1.0, GammaLambda: 2e-5, GammaMu: 2e-5, etaY: 0.9, etaDual: 0.03 };

export class GdaState {
  readonly T: number;
  readonly S: number;
  iter = 0;
  y: Float64Array;
  lamRes: Float64Array;
  lamNn: Float64Array;
  mu0: Float64Array;
  muf: Float64Array;
  private gy: Float64Array;
  private glr: Float64Array;
  private gln: Float64Array;
  private gm0: Float64Array;
  private gmf: Float64Array;
  private P: Float64Array; // [u][s][s2]
  private c: Float64Array; // [s][u]
  private budgets: [number, number];
  private m0: Float64Array;

  constructor(
    model: Model,
    readonly hyper: GdaHyper = DEFAULT_HYPER,
  ) {
    this.T = model.T;
    this.S = model.S;
    const n = this.T * this.S * U;
    this.y = new Float64Array(n);
    this.lamRes = new Float64Array(2 * this.T);
    this.lamNn = new Float64Array(n);
    this.mu0 = new Float64Array(this.S);
    this.muf = new Float64Array((this.T - 1) * this.S);
    this.gy = new Float64Array(n);
    this.glr = new Float64Array(2 * this.T);
    this.gln = new Float64Array(n);
    this.gm0 = new Float64Array(this.S);
    this.gmf = new Float64Array((this.T - 1) * this.S);
    const S = this.S;
    this.P = new Float64Array(U * S * S);
    for (let u = 0; u < U; u++) for (let s = 0; s < S; s++) for (let s2 = 0; s2 < S; s2++) this.P[(u * S + s) * S + s2] = model.P[u][s][s2];
    this.c = new Float64Array(S * U);
    for (let s = 0; s < S; s++) for (let u = 0; u < U; u++) this.c[s * U + u] = model.c[s][u];
    this.budgets = [model.budgets[0], model.budgets[1]];
    this.m0 = Float64Array.from(model.m0);
  }

  /** Run `n` iterations. */
  run(n: number): void {
    const { T, S, P, c, y, lamRes, lamNn, mu0, muf, gy, glr, gln, gm0, gmf } = this;
    const { GammaY: Gy, GammaLambda: Gl, GammaMu: Gm, etaY, etaDual } = this.hyper;
    const Gn = Gl;
    const [alpha, beta] = this.budgets;
    for (let it = 0; it < n; it++) {
      // gradients at the current point
      for (let t = 0; t < T; t++) {
        const lr0 = lamRes[t];
        const lr1 = lamRes[T + t];
        for (let s = 0; s < S; s++) {
          const muHere = t === 0 ? mu0[s] : muf[(t - 1) * S + s];
          for (let u = 0; u < U; u++) {
            const i = (t * S + s) * U + u;
            let g = c[s * U + u] + Gy * y[i] - lamNn[i] + lr0 * ADMIT[u] + lr1 * HIGH[u] + muHere;
            if (t < T - 1) {
              let acc = 0;
              const base = (u * S + s) * S;
              const mb = t * S;
              for (let s2 = 0; s2 < S; s2++) acc += P[base + s2] * muf[mb + s2];
              g -= acc;
            }
            gy[i] = g;
            gln[i] = -y[i] - Gn * lamNn[i];
          }
        }
      }
      for (let t = 0; t < T; t++) {
        let use0 = 0;
        let use1 = 0;
        for (let s = 0; s < S; s++) {
          const b = (t * S + s) * U;
          use0 += y[b + 2] + y[b + 3];
          use1 += y[b + 1] + y[b + 3];
        }
        glr[t] = use0 - alpha - Gl * lamRes[t];
        glr[T + t] = use1 - beta - Gl * lamRes[T + t];
      }
      for (let s = 0; s < S; s++) {
        const b = s * U;
        gm0[s] = y[b] + y[b + 1] + y[b + 2] + y[b + 3] - this.m0[s] - Gm * mu0[s];
      }
      for (let t = 1; t < T; t++) {
        for (let s = 0; s < S; s++) {
          const b = (t * S + s) * U;
          let acc = y[b] + y[b + 1] + y[b + 2] + y[b + 3];
          let inflow = 0;
          for (let s2 = 0; s2 < S; s2++) {
            const bp = ((t - 1) * S + s2) * U;
            for (let u = 0; u < U; u++) inflow += y[bp + u] * P[(u * S + s2) * S + s];
          }
          acc -= inflow;
          gmf[(t - 1) * S + s] = acc - Gm * muf[(t - 1) * S + s];
        }
      }
      // simultaneous update, multipliers of inequalities projected on the nonnegative orthant
      for (let i = 0; i < y.length; i++) {
        y[i] -= etaY * gy[i];
        const v = lamNn[i] + etaDual * gln[i];
        lamNn[i] = v > 0 ? v : 0;
      }
      for (let i = 0; i < lamRes.length; i++) {
        const v = lamRes[i] + etaDual * glr[i];
        lamRes[i] = v > 0 ? v : 0;
      }
      for (let s = 0; s < S; s++) mu0[s] += etaDual * gm0[s];
      for (let i = 0; i < muf.length; i++) muf[i] += etaDual * gmf[i];
    }
    this.iter += n;
  }

  /** The current plan as nested arrays y[t][s][u]. */
  plan(): number[][][] {
    const { T, S, y } = this;
    return Array.from({ length: T }, (_, t) => Array.from({ length: S }, (_, s) => Array.from({ length: U }, (_, u) => y[(t * S + s) * U + u])));
  }

  /** The current prices lam[l][t] (0 = cars, 1 = attendants). */
  prices(): number[][] {
    return [Array.from(this.lamRes.slice(0, this.T)), Array.from(this.lamRes.slice(this.T))];
  }
}
