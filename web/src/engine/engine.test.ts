/**
 * The browser engine against the Python implementation (web/src/data/reference.json, written by the `web` stage from
 * lp_learner): kernels, backward induction, the LP's induced policy, the large-N evaluation and GDA must agree.
 * The simulator is checked statistically against the transition probabilities.
 */
import { describe, expect, it } from 'vitest';
import ref from '../data/reference.json';
import { buildModel, perPeriodDeparture, transitionKernel, type Params } from './model';
import { deterministicPolicy, pricedSolution, selfishSolution } from './dp';
import { conditionalPolicy } from './policy';
import { fluidEvaluate } from './fluid';
import { GdaState } from './gda';
import { Plaza, projectionController, repairCounts, systematicRound, tableController } from './sim';
import { Rng } from './rng';

const params = ref.params as Params;
const model = buildModel(params);

function maxAbsDiff(a: unknown, b: unknown): number {
  if (Array.isArray(a) && Array.isArray(b)) {
    expect(a.length).toBe(b.length);
    return a.reduce((m: number, x, i) => Math.max(m, maxAbsDiff(x, b[i])), 0);
  }
  return Math.abs((a as number) - (b as number));
}

describe('model', () => {
  it('kernels match Python', () => {
    expect(maxAbsDiff(transitionKernel(params.K, params.p, params.bLow), ref.kernelLow)).toBeLessThan(1e-12);
    expect(maxAbsDiff(transitionKernel(params.K, params.p, params.bHigh), ref.kernelHigh)).toBeLessThan(1e-12);
    expect(maxAbsDiff(transitionKernel(params.K, params.p, ref.kernelMid.q), ref.kernelMid.P)).toBeLessThan(1e-12);
    expect(maxAbsDiff(model.P, ref.P)).toBeLessThan(1e-12);
    expect(maxAbsDiff(model.c, ref.c)).toBeLessThan(1e-12);
  });

  it('per-period departure probability is the kernel mean', () => {
    const q = params.bHigh;
    const r = perPeriodDeparture(params.p, q);
    const row = transitionKernel(params.K, params.p, q)[1][3]; // 4 cars to serve
    const mean = row.reduce((acc, v, s) => acc + v * s, 0);
    expect(mean).toBeCloseTo(4 * (1 - r), 12);
  });
});

describe('backward induction', () => {
  it('selfish policy and values match Python for both tie-breaks', () => {
    for (const tie of ['no_request', 'request'] as const) {
      const sol = selfishSolution(model, tie);
      const r = (ref as Record<string, any>)[`selfish_${tie}`];
      expect(maxAbsDiff(sol.V, r.V)).toBeLessThan(1e-9);
      expect(sol.u).toEqual(r.u);
    }
  });

  it('price-taking policy with the LP prices matches Python', () => {
    expect(pricedSolution(model, ref.lpLam, 'no_request').u).toEqual(ref.priced_no_request.u);
  });
});

describe('policies and large-N evaluation', () => {
  it('conditional policy of the LP plan matches Python', () => {
    expect(maxAbsDiff(conditionalPolicy(ref.lpY), ref.piLp)).toBeLessThan(1e-9);
  });

  it('fluid costs, masses and grants match Python under every rule', () => {
    const sol = selfishSolution(model, 'no_request');
    for (const mech of ['M0', 'M1', 'M2'] as const) {
      const fl = fluidEvaluate(model, deterministicPolicy(sol.u), mech);
      const r = (ref.fluid as Record<string, any>)[`selfish_${mech}`];
      expect(fl.cost).toBeCloseTo(r.cost, 8);
      expect(maxAbsDiff(fl.mass, r.mass)).toBeLessThan(1e-10);
      expect(maxAbsDiff(fl.grantA, r.grantA)).toBeLessThan(1e-10);
      expect(maxAbsDiff(fl.qHigh, r.qHigh)).toBeLessThan(1e-10);
    }
    const lp = fluidEvaluate(model, conditionalPolicy(ref.lpY), 'M1');
    expect(lp.cost).toBeCloseTo(ref.fluid.lp_M1.cost, 6);
    expect(lp.cost).toBeCloseTo(ref.lpValue, 3);
  });
});

describe('GDA', () => {
  it('reproduces the Python iterates from a zero start', () => {
    const g = new GdaState(model, {
      GammaY: ref.gda.GammaY, GammaLambda: ref.gda.GammaLambda, GammaMu: ref.gda.GammaMu, etaY: ref.gda.etaY, etaDual: ref.gda.etaDual,
    });
    g.run(ref.gda.nIter);
    expect(maxAbsDiff(g.plan(), ref.gda.y)).toBeLessThan(1e-8);
    expect(maxAbsDiff(g.prices(), ref.gda.lamRes)).toBeLessThan(1e-8);
  });
});

describe('simulator', () => {
  it('one-period transitions follow the kernel', () => {
    const N = 4000;
    const plaza = new Plaza(model, N, 'rationing', 7);
    plaza.states = new Array(N).fill(3);
    const always = tableController('test', Array.from({ length: params.T }, () => new Array(params.K + 1).fill(3))); // (1,1)
    // with enough budget everyone gets both resources
    const big = buildModel({ ...params, alpha: 1, beta: 1 });
    const p2 = new Plaza(big, N, 'rationing', 7);
    p2.states = plaza.states.slice();
    const out = p2.step(always);
    const freq = new Array(params.K + 1).fill(0);
    for (const l of out.lanes) freq[l.sNext] += 1 / N;
    const row = big.P[3][3];
    for (let s = 0; s <= params.K; s++) expect(Math.abs(freq[s] - row[s])).toBeLessThan(0.03);
  });

  it('rounding keeps row sums and repair meets the budgets', () => {
    const rng = new Rng(1);
    const z = [[1.4, 0.6, 2.2, 0.8], [0.25, 0.25, 0.25, 0.25], [0, 0, 3, 0]];
    const n = systematicRound(z, rng);
    n.forEach((row, s) => expect(row.reduce((a, b) => a + b, 0)).toBe(Math.round(z[s].reduce((a, b) => a + b, 0))));
    const pi = z.map((row) => row.map((v) => v / row.reduce((a, b) => a + b, 0)));
    const { n: rep } = repairCounts(n, pi, 3, 1);
    expect(rep.reduce((a, r) => a + r[2] + r[3], 0)).toBeLessThanOrEqual(3);
    expect(rep.reduce((a, r) => a + r[1] + r[3], 0)).toBeLessThanOrEqual(1);
  });

  it('the coordinator never exceeds the budgets', () => {
    const pi = conditionalPolicy(ref.lpY);
    const plaza = new Plaza(model, 37, 'rationing', 3);
    const ctrl = projectionController(pi, model.budgets);
    for (let k = 0; k < 50; k++) {
      const out = plaza.step(ctrl);
      expect(out.lanes.reduce((a, l) => a + l.aReq, 0)).toBeLessThanOrEqual(out.capA);
      expect(out.lanes.reduce((a, l) => a + l.bReq, 0)).toBeLessThanOrEqual(out.capB);
    }
  });
});
