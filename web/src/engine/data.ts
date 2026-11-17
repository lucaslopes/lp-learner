/**
 * The data exported by the `web` stage (web/src/data/model.json): base parameters, exact LP plans on a grid of
 * budgets, and the recorded results the page quotes. Everything else on the page is computed live by the engine.
 */
import raw from '../data/model.json';
import { buildModel, type Model, type Params } from './model';
import { conditionalPolicy } from './policy';
import { deterministicPolicy, pricedSolution, selfishSolution, type DpSolution } from './dp';
import { fluidEvaluate, type FluidResult } from './fluid';

export interface LpPlan {
  alpha: number;
  beta: number;
  value: number;
  y: number[][][];
  lam: number[][];
}

interface CostCi {
  mean: number;
  ci: number;
}

export interface Results {
  lpValue: number;
  replications: number;
  source: string;
  main: { N: number; coord: CostCi; rationing: CostCi & { diff: number; diffCi: number; pct: number }; congestion: CostCi & { diff: number; diffCi: number; pct: number } }[];
  detail: Record<'coord' | 'selfishRationing' | 'selfishCongestion' | 'priceTaking', CostCi & {
    carRequestsPerCar: number; attRequestsPerAtt: number; overloadCars: number; overloadAtt: number; fullLanes: number; lostCars: number;
  }>;
  dilemma: Record<'rationing' | 'congestion', {
    collective: { diff: number; ci: number };
    deviations: Record<'neverAttendant' | 'followCoordinator' | 'neverGate', { diff: number; ci: number }>;
    verdict: string;
  }>;
  whatif: { scenario: string; label: string; coord: CostCi; rationing: number; congestion: number; lpValue: number }[];
  coordinators: { label: string; costs: Record<string, number> }[];
  fluidLostCumulative: Record<'lp' | 'selfish_M1' | 'selfish_M2', number[]>;
  learning: Record<'lp' | 'gda' | 'sgda', { repairsPerPeriod: number; massFinal: number }>;
  sensitivity: { scenario: string; N: number; rationingPct: number }[];
  unavoidableMissed: number;
  carsPerBooth: number;
}

export const BASE: Params = raw.params as Params;
export const RESULTS: Results = raw.results as unknown as Results;
export const LP_GRID: LpPlan[] = raw.lpGrid as LpPlan[];
export const ALPHAS = [...new Set(LP_GRID.map((g) => g.alpha))].sort((a, b) => a - b);
export const BETAS = [...new Set(LP_GRID.map((g) => g.beta))].sort((a, b) => a - b);

export function lpPlan(alpha: number, beta: number): LpPlan {
  const hit = LP_GRID.find((g) => Math.abs(g.alpha - alpha) < 1e-9 && Math.abs(g.beta - beta) < 1e-9);
  if (!hit) throw new Error(`no LP plan for alpha=${alpha}, beta=${beta}`);
  return hit;
}

export function snapToGrid(values: number[], x: number): number {
  return values.reduce((best, v) => (Math.abs(v - x) < Math.abs(best - x) ? v : best), values[0]);
}

/** Everything the page derives from one choice of budgets. */
export interface Scenario {
  params: Params;
  model: Model;
  lp: LpPlan;
  piLp: number[][][];
  selfish: DpSolution;
  priced: DpSolution;
  fluid: { lp: FluidResult; selfishRationing: FluidResult; selfishCongestion: FluidResult; price: FluidResult };
}

const cache = new Map<string, Scenario>();

export function scenario(alpha = BASE.alpha, beta = BASE.beta): Scenario {
  const key = `${alpha}|${beta}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const params = { ...BASE, alpha, beta };
  const model = buildModel(params);
  const lp = lpPlan(alpha, beta);
  const piLp = conditionalPolicy(lp.y);
  const selfish = selfishSolution(model, 'no_request');
  const priced = pricedSolution(model, lp.lam, 'no_request');
  const sc: Scenario = {
    params, model, lp, piLp, selfish, priced,
    fluid: {
      lp: fluidEvaluate(model, piLp, 'M1'),
      selfishRationing: fluidEvaluate(model, deterministicPolicy(selfish.u), 'M1'),
      selfishCongestion: fluidEvaluate(model, deterministicPolicy(selfish.u), 'M2'),
      price: fluidEvaluate(model, deterministicPolicy(priced.u), 'M1'),
    },
  };
  cache.set(key, sc);
  return sc;
}

export const fmt = (x: number, digits = 1) => (Math.abs(x) < 0.5 * 10 ** -digits ? (0).toFixed(digits) : x.toFixed(digits));
export const signed = (x: number, digits = 1) => (x >= 0 ? '+' : '−') + fmt(Math.abs(x), digits);
