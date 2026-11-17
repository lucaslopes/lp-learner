/**
 * The playground: the whole story in one app. The plaza runs rush hour after rush hour while the chosen policy decides;
 * with "AI coordinator (learning)", the AI learns its plan live (GDA in a web worker) and the plaza always uses the
 * latest learned plan, so the cost per rush hour falls as it learns. Every panel links back to the section that
 * explains it.
 */
import { useMemo, useState } from 'react';
import { PlazaStage } from '../components/PlazaStage';
import { PiePlot } from '../components/PiePlot';
import { Button, Explain, Slider, Stat, Toggle } from '../components/controls';
import { LineChart } from '../components/charts';
import { COORD_COLOR, CONGESTION_COLOR, PRICE_COLOR, SELFISH_COLOR } from '../components/art';
import { ALPHAS, BASE, BETAS, scenario, fmt } from '../engine/data';
import { capacities } from '../engine/model';
import { deterministicPolicy } from '../engine/dp';
import { fluidEvaluate, occupation, type Mechanism } from '../engine/fluid';
import { conditionalPolicy } from '../engine/policy';
import { projectionController, tableController, type Controller, type PeriodOutcome, type Rule } from '../engine/sim';
import { useGda } from '../hooks/useGda';

type PolicyKey = 'learning' | 'solved' | 'selfish' | 'price';
const POLICY_LABEL: Record<PolicyKey, string> = {
  learning: 'AI coordinator (learning)',
  solved: 'AI coordinator (solved LP)',
  selfish: 'selfish booths',
  price: 'price-taking booths',
};
const POLICY_COLOR: Record<PolicyKey, string> = { learning: '#333', solved: COORD_COLOR, selfish: SELFISH_COLOR, price: PRICE_COLOR };

export function Playground() {
  const [policy, setPolicy] = useState<PolicyKey>('learning');
  const [rule, setRule] = useState<Rule>('rationing');
  const [N, setN] = useState(24);
  const [alpha, setAlpha] = useState(BASE.alpha);
  const [beta, setBeta] = useState(BASE.beta);
  const [speed, setSpeed] = useState(2);
  const [playing, setPlaying] = useState(true);
  const [resetKey, setResetKey] = useState(0);
  const [last, setLast] = useState<PeriodOutcome | null>(null);
  const [hours, setHours] = useState<{ k: number; cost: number; policy: PolicyKey }[]>([]);

  const sc = useMemo(() => scenario(alpha, beta), [alpha, beta]);
  const gda = useGda(sc.params);
  const mech: Mechanism = rule === 'rationing' ? 'M1' : 'M2';
  const learnedPi = useMemo(() => (gda.snap ? conditionalPolicy(gda.snap.y) : null), [gda.snap]);

  const controller: Controller = useMemo(() => {
    if (policy === 'solved') return projectionController(sc.piLp, sc.model.budgets);
    if (policy === 'selfish') return tableController('selfish', sc.selfish.u);
    if (policy === 'price') return tableController('price', sc.priced.u);
    return learnedPi ? projectionController(learnedPi, sc.model.budgets) : tableController('idle', sc.selfish.u.map((r) => r.map(() => 0)));
  }, [policy, sc, learnedPi]);

  const policyPi = useMemo(() => {
    if (policy === 'solved') return sc.piLp;
    if (policy === 'selfish') return deterministicPolicy(sc.selfish.u);
    if (policy === 'price') return deterministicPolicy(sc.priced.u);
    return learnedPi;
  }, [policy, sc, learnedPi]);
  const fluidNow = useMemo(() => (policyPi ? fluidEvaluate(sc.model, policyPi, mech) : null), [policyPi, sc, mech]);
  const selfishFluid = rule === 'rationing' ? sc.fluid.selfishRationing : sc.fluid.selfishCongestion;
  const pies = useMemo(() => {
    if (policy === 'solved') return sc.lp.y;
    if (policy === 'learning') return gda.snap && gda.snap.iter > 0 ? gda.snap.y : null;
    return fluidNow && policyPi ? occupation(fluidNow, policyPi, mech, true) : null;
  }, [policy, sc, gda.snap, fluidNow, policyPi, mech]);

  const restart = () => {
    setHours([]);
    setLast(null);
    setResetKey((k) => k + 1);
  };
  // A new policy starts on a fresh plaza, so that no rush hour mixes two policies. The history is kept for the chart.
  const choosePolicy = (p: PolicyKey) => {
    setPolicy(p);
    setLast(null);
    setResetKey((k) => k + 1);
  };
  const changeAndRestart = <T,>(set: (v: T) => void) => (v: T) => {
    set(v);
    setHours([]);
    setLast(null);
  };

  // Average over the rush hours played with the current policy since it was chosen, with a 95% range.
  const segment: number[] = [];
  for (let i = hours.length - 1; i >= 0 && hours[i].policy === policy; i--) segment.push(hours[i].cost);
  const avg = segment.length ? segment.reduce((a, c) => a + c, 0) / segment.length : null;
  const sd = segment.length > 1 && avg != null ? Math.sqrt(segment.reduce((a, c) => a + (c - avg) ** 2, 0) / (segment.length - 1)) : null;
  const half = sd != null ? (1.96 * sd) / Math.sqrt(segment.length) : null;
  const [capA, capB] = capacities(N, alpha, beta);
  const tSeries = (['learning', 'solved', 'selfish', 'price'] as PolicyKey[])
    .map((p) => ({ label: POLICY_LABEL[p], color: POLICY_COLOR[p], marker: true, line: false, points: hours.filter((h) => h.policy === p).map((h) => [h.k, Math.min(h.cost, 700)] as [number, number]) }))
    .filter((s) => s.points.length);
  const t = Array.from({ length: BASE.T }, (_, k) => k);

  return (
    <section id="playground" className="playground">
      <header className="section-head">
        <span className="section-number">★</span>
        <h2>The playground</h2>
        <p className="kicker">
          Everything at once. Pick who decides, change the budgets, and watch the AI compute its plan while the cars roll. Each panel has a link back to the part of the story that explains it.
        </p>
      </header>

      <div className="pg-controls">
        <Toggle label="Who decides?" value={policy} onChange={choosePolicy} options={(Object.keys(POLICY_LABEL) as PolicyKey[]).map((p) => ({ value: p, label: POLICY_LABEL[p] }))} />
        <Toggle label="Too many requests?" value={rule} onChange={changeAndRestart(setRule)} options={[{ value: 'rationing', label: 'lottery' }, { value: 'congestion', label: 'share attendants' }]} />
        <Slider label="booths N =" value={N} min={4} max={30} onChange={changeAndRestart(setN)} />
        <Slider label="cars per booth α =" value={alpha} min={0} max={1} marks={ALPHAS} onChange={changeAndRestart(setAlpha)} format={(v) => v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')} />
        <Slider label="attendants per booth β =" value={beta} min={0} max={1} marks={BETAS} onChange={changeAndRestart(setBeta)} format={(v) => v.toFixed(2)} />
        <Toggle label="speed" value={speed} onChange={setSpeed} options={[{ value: 0.5, label: '½×' }, { value: 1, label: '1×' }, { value: 2, label: '2×' }, { value: 4, label: '4×' }, { value: 16, label: 'turbo' }]} />
        <div className="pg-buttons">
          <Button kind="primary" onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause plaza' : 'Play plaza'}</Button>
          <Button onClick={restart}>Restart plaza</Button>
        </div>
      </div>

      <div className="pg-grid">
        <div className="panel panel-plaza">
          <div className="panel-head">
            <h3>The plaza <small>({POLICY_LABEL[policy]})</small></h3>
            <Explain to="many">the plaza and its budgets</Explain>
          </div>
          <PlazaStage
            model={sc.model} N={N} rule={rule} controller={controller} playing={playing} speed={speed} resetKey={resetKey}
            showRequests={N <= 16} onPeriod={setLast}
            onRushHour={(cost) => setHours((h) => [...h.slice(-79), { k: (h.length ? h[h.length - 1].k : 0) + 1, cost, policy }])}
            height="auto"
          />
          {last && (
            <div className="stats-row">
              <Stat label="this period: cars asked / available" value={`${last.requestsA} / ${last.capA}`} tone={last.requestsA > last.capA ? 'warn' : undefined} />
              <Stat label="attendants asked / available" value={`${last.requestsB} / ${last.capB}`} tone={last.requestsB > last.capB ? 'warn' : undefined} />
              <Stat label="cars lost this period" value={`${last.carsLost}`} sub={<Explain to="costs">missed tolls</Explain>} />
            </div>
          )}
        </div>

        <div className="panel panel-plan">
          <div className="panel-head">
            <h3>{policy === 'learning' ? "The AI's plan, as it learns" : policy === 'solved' ? "The AI's plan (solved)" : 'What the booths do (many booths)'}</h3>
            <Explain to={policy === 'learning' ? 'learning' : policy === 'price' ? 'prices' : 'coordinator'}>the pie plot</Explain>
          </div>
          {pies ? <PiePlot y={pies} compact legend /> : <div className="empty-plan">The AI has not learned anything yet: with an empty plan it opens no gate at all. Press <b>Start learning</b>.</div>}
          {policy === 'learning' && (
            <>
              <div className="demo-controls">
                {gda.running ? <Button onClick={gda.pause}>Pause learning</Button> : <Button kind="primary" onClick={gda.run} disabled={gda.snap?.done}>{gda.snap && gda.snap.iter > 0 ? 'Keep learning' : 'Start learning'}</Button>}
                <Button kind="ghost" onClick={gda.reset}>Forget</Button>
                <div className="progress"><i style={{ width: `${100 * (gda.snap ? gda.snap.iter / gda.maxIter : 0)}%` }} /><span>{gda.snap ? `${(gda.snap.iter / 1000).toFixed(0)}k rounds` : '…'}</span></div>
              </div>
              {gda.snap && (
                <LineChart
                  title="Prices: learned (solid) and solved (dashed)"
                  xTicks={t}
                  height={170}
                  yDomain={[0, 70]}
                  series={[
                    { label: 'attendant, learned', color: COORD_COLOR, marker: true, points: t.map((k) => [k, gda.snap!.lam[1][k]] as [number, number]) },
                    { label: 'car, learned', color: PRICE_COLOR, marker: true, points: t.map((k) => [k, gda.snap!.lam[0][k]] as [number, number]) },
                    { label: 'attendant, solved', color: '#7aa9cf', dash: '4 4', points: t.map((k) => [k, sc.lp.lam[1][k]] as [number, number]) },
                  ]}
                />
              )}
            </>
          )}
        </div>

        <div className="panel panel-score">
          <div className="panel-head">
            <h3>Scoreboard</h3>
            <Explain to="verdict">what the big simulations say</Explain>
          </div>
          <div className="stats-row">
            <Stat label="rush hours played" value={hours.length} />
            <Stat
              label={`average cost per booth (${segment.length} rush hour${segment.length === 1 ? '' : 's'})`}
              value={avg != null ? fmt(avg, 1) : '…'}
              sub={half != null ? `± ${fmt(half, 1)} (95% range) · ${POLICY_LABEL[policy]}` : POLICY_LABEL[policy]}
            />
            <Stat tone="coord" label="LP lower bound (many booths)" value={fmt(sc.lp.value, 1)} sub={<Explain to="coordinator">why a bound?</Explain>} />
            <Stat
              label="this policy with many booths (computed live)"
              value={fluidNow ? fmt(fluidNow.cost, 1) : '…'}
              sub={
                <>
                  {rule === 'rationing' ? 'Excess requests: lottery' : 'Excess cars: lottery; attendants: shared'}
                  {(policy === 'learning' || policy === 'solved') && <><br />The plaza repairs the plan instead.</>}
                </>
              }
            />
            <Stat tone="selfish" label="selfish booths with many booths" value={fmt(selfishFluid.cost, 1)} sub={rule === 'rationing' ? 'lottery' : 'shared attendants'} />
          </div>
          <LineChart
            title="Cost per booth, rush hour after rush hour"
            xLabel="rush hour"
            yLabel="cost per booth"
            height={240}
            width={1100}
            series={tSeries.length ? tSeries : [{ label: 'no rush hour finished yet', color: '#ccc', points: [[0, sc.lp.value]] }]}
            hLines={[
              { y: sc.lp.value, label: 'LP lower bound', color: COORD_COLOR, below: true },
              { y: selfishFluid.cost, label: 'selfish, many booths', color: rule === 'rationing' ? SELFISH_COLOR : CONGESTION_COLOR },
            ]}
            note={`One rush hour can be lucky or unlucky, so compare averages, not single dots. A small plaza also does a bit worse than the many-booth lines: with ${N} booths it gets ${capA} cars and ${capB} attendants (whole numbers), and the coordinator's rounding matters more. The dashed lines are many-booth values. Costs above 700 are drawn at 700.`}
          />
        </div>
      </div>
    </section>
  );
}
