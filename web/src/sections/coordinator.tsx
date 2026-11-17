/** Part 4: the AI coordinator. Plans with fractions (the LP), prices, from the plan to orders, and learning the plan. */
import { useMemo, useState } from 'react';
import { Scrolly } from '../components/Scrolly';
import { TeX } from '../components/TeX';
import { PiePlot } from '../components/PiePlot';
import { Button, Slider, Stat, Toggle } from '../components/controls';
import { BarChart, LineChart } from '../components/charts';
import { ACTION_WORDS, COORD_COLOR, PIE_COLORS, PRICE_COLOR, SELFISH_COLOR } from '../components/art';
import { BASE, RESULTS, scenario, fmt } from '../engine/data';
import { deterministicPolicy } from '../engine/dp';
import { fluidEvaluate, occupation } from '../engine/fluid';
import { capacities } from '../engine/model';
import { conditionalPolicy } from '../engine/policy';
import { Plaza, projectionController, type RoundingTrace } from '../engine/sim';
import { useGda } from '../hooks/useGda';

const sc = scenario();
const selfishPi = deterministicPolicy(sc.selfish.u);
const occAsk = occupation(sc.fluid.selfishRationing, selfishPi, 'M1', false);
const occGet = occupation(sc.fluid.selfishRationing, selfishPi, 'M1', true);

// ---------------------------------------------------------------------------------------------------------------------
// 9. Thinking in fractions: the LP

function PlanDemo({ step }: { step: number }) {
  const [view, setView] = useState<'plan' | 'ask' | 'get'>('plan');
  const [t, setT] = useState(4);
  const y = view === 'plan' ? sc.lp.y : view === 'ask' ? occAsk : occGet;
  const carsUsed = y[t].reduce((acc, row) => acc + row[2] + row[3], 0);
  const attUsed = y[t].reduce((acc, row) => acc + row[1] + row[3], 0);
  return (
    <div className="demo">
      {step >= 5 && (
        <div className="demo-controls">
          <Toggle value={view} onChange={setView} options={[
            { value: 'plan', label: "the coordinator's plan" },
            { value: 'ask', label: 'selfish: what they ask' },
            { value: 'get', label: 'selfish: what they get' },
          ]} />
        </div>
      )}
      <PiePlot y={y} highlightT={step >= 2 && step < 5 ? t : null} />
      {step >= 2 && step < 5 && (
        <div className="demo-controls">
          <Slider label="period t =" value={t} min={0} max={BASE.T - 1} onChange={setT} />
          <span className={`pill ${carsUsed <= BASE.alpha + 1e-6 ? 'ok' : 'bad'}`}>cars: <b>{fmt(carsUsed, 2)}</b> ≤ α = {BASE.alpha}</span>
          <span className={`pill ${attUsed <= BASE.beta + 1e-6 ? 'ok' : 'bad'}`}>attendants: <b>{fmt(attUsed, 2)}</b> ≤ β = {BASE.beta}</span>
        </div>
      )}
      {step === 4 && (
        <div className="stats-row">
          <Stat tone="coord" label="LP value: the yardstick" value={fmt(sc.lp.value, 1)} sub="cost per booth; no coordinator can do better" />
          <Stat tone="selfish" label="selfish booths, many booths, lottery" value={fmt(sc.fluid.selfishRationing.cost, 1)} sub="computed live in your browser" />
        </div>
      )}
    </div>
  );
}
export function SectionPlan() {
  return (
    <Scrolly
      id="coordinator"
      number="9"
      title="The AI coordinator thinks in fractions"
      kicker="A central program sees every lane and decides for everybody. Here is its trick."
      steps={[
        {
          key: 'a',
          body: (
            <p>
              Enter the <b>AI coordinator</b>: a program that sees every lane and tells each booth what to do. ("AI" here means a planner that uses a model of the plaza, not a chatbot.) It cannot use Bellman's equation on <TeX>{'6^N'}</TeX> states. So it changes the question.
            </p>
          ),
        },
        {
          key: 'b',
          body: (
            <>
              <p>With many booths, it does not matter <i>which</i> booth has 3 cars, only <i>how many</i> do. The coordinator plans with fractions:</p>
              <TeX block>{'y_{s,a,b}(t) = \\text{share of booths with } s \\text{ cars doing } (a,b) \\text{ at period } t'}</TeX>
              <p className="small-note">(More precisely the <i>expected</i> share, because luck changes the real share a little.)</p>
              <p>That is exactly what the <b>pie plot</b> shows. Each pie is one period and one lane length. Its size is the share of booths there, and its slices are what they do.</p>
            </>
          ),
        },
        {
          key: 'c',
          body: (
            <>
              <p>The fractions must obey bookkeeping rules: everyone starts empty; booths move between states with the probabilities <TeX>{"P(s'\\mid s,a,b)"}</TeX>; and, <b>on average</b>, the budgets hold:</p>
              <TeX block>{'\\sum_{s,b} y_{s,1,b}(t) \\le \\alpha, \\qquad \\sum_{s,a} y_{s,a,1}(t) \\le \\beta'}</TeX>
              <p>Move the period slider: the plan uses at most its budget in every column.</p>
            </>
          ),
        },
        {
          key: 'd',
          body: (
            <>
              <p>Find the fractions with the smallest total cost:</p>
              <TeX block>{'\\min_{y \\ge 0}\\; \\sum_{t}\\sum_{s,a,b} c(s,a,b)\\, y_{s,a,b}(t) \\quad \\text{subject to the rules}'}</TeX>
              <p>This is a <b>linear program</b> (LP) with just 240 numbers, however many booths there are. A computer solves it in milliseconds.</p>
            </>
          ),
        },
        {
          key: 'e',
          body: (
            <p>
              Its answer, <b>{fmt(sc.lp.value, 2)}</b> per booth, is a <b>lower bound</b>. Any plaza policy that respects the budgets <i>every</i> time also respects them on average, so it is one of the fractions the LP considered. No coordinator, however clever, can do better.
            </p>
          ),
        },
        {
          key: 'f',
          body: (
            <p>
              Now compare, using the buttons above the pies. In the coordinator's plan, the dark green slices (gate open <i>and</i> an attendant) sit on the <b>top pie</b> of each early column: attendants go to the longest lanes, where they help most. The selfish booths ask for everything everywhere. After the lottery, about a quarter of the booths in <i>every</i> state get an attendant, even the empty lanes.
            </p>
          ),
        },
      ]}
      render={(step) => <PlanDemo step={step} />}
    />
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// 10. Prices

function PricesDemo({ step }: { step: number }) {
  const t = Array.from({ length: BASE.T }, (_, k) => k);
  const pricedOcc = useMemo(() => occupation(sc.fluid.price, deterministicPolicy(sc.priced.u), 'M1', true), []);
  return (
    <div className="demo">
      {step < 3 ? (
        <LineChart
          title="The coordinator's prices"
          xLabel="period t"
          yLabel="price (cost units)"
          xTicks={t}
          series={[
            { label: 'one more attendant', color: COORD_COLOR, marker: true, points: t.map((k) => [k, sc.lp.lam[1][k]]) },
            { label: 'one more car', color: PRICE_COLOR, marker: true, points: t.map((k) => [k, sc.lp.lam[0][k]]) },
          ]}
        />
      ) : (
        <>
          <PiePlot y={pricedOcc} title="Price-taking booths (many booths, lottery)" legend={false} compact />
          <BarChart
            title="Cost per booth, many booths (computed live)"
            groups={['coordinator (LP)', 'selfish', 'price-taking']}
            series={[{ label: 'cost per booth', color: '#555', values: [sc.lp.value, sc.fluid.selfishRationing.cost, sc.fluid.price.cost] }]}
            format={(v) => v.toFixed(0)}
            height={190}
          />
        </>
      )}
    </div>
  );
}

export function SectionPrices() {
  const maxAtt = Math.max(...sc.lp.lam[1]);
  const priceRefused = sc.fluid.price.requested[0].flatMap((r, t) => (r < 1e-9 ? [t] : []));
  return (
    <Scrolly
      id="prices"
      number="10"
      title="Every budget has a price"
      kicker="Solving the LP also tells us how valuable one more attendant or one more car would be."
      steps={[
        {
          key: 'a',
          body: (
            <p>
              Each budget rule of the LP comes with a hidden number, its <b>price</b> (mathematicians say <i>dual variable</i>). <TeX>{'\\lambda_{\\text{att}}(t)'}</TeX> says how much the whole plaza's cost would drop with one more attendant in period <TeX>t</TeX>.
            </p>
          ),
        },
        {
          key: 'b',
          body: (
            <p>
              The attendant's price climbs to about <b>{maxAtt.toFixed(0)}</b> in the middle of the rush hour, when lanes are long, and drops to zero at the end, when there is no future to save. A car is worth much less: a car admitted now usually takes the place of a later one.
            </p>
          ),
        },
        {
          key: 'c',
          body: (
            <>
              <p>Tempting idea: post the prices and let every booth decide by itself, paying for what it uses:</p>
              <TeX block>{'c(s,a,b) + \\lambda_{\\text{car}}(t)\\, a + \\lambda_{\\text{att}}(t)\\, b'}</TeX>
              <p>Each booth then solves its own small MDP. No coordinator needed?</p>
            </>
          ),
        },
        {
          key: 'd',
          body: (
            <p>
              It goes badly wrong. At the exact prices, booths are often <b>indifferent</b>: several actions cost the same. The coordinator splits such ties in precise proportions, but a booth alone cannot know its share. Our booths all break ties the same way, so they all refuse the platoon in periods {priceRefused.join(' and ')}. The pies show it: no green at the start.
            </p>
          ),
        },
        {
          key: 'e',
          body: (
            <p>
              Lesson: prices tell booths what things are <i>worth</i>; somebody still has to decide <i>who gets them</i>. (In our big simulations, price-taking booths cost {fmt(RESULTS.detail.priceTaking.mean, 1)} per booth, {fmt((100 * (RESULTS.detail.priceTaking.mean - RESULTS.detail.coord.mean)) / RESULTS.detail.coord.mean, 1)}% more than the coordinator.)
            </p>
          ),
        },
      ]}
      render={(step) => <PricesDemo step={step} />}
    />
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// 11. From the plan to orders

function RoundingDemo({ step }: { step: number }) {
  const N = 20;
  const [capA, capB] = capacities(N, BASE.alpha, BASE.beta);
  const [state, setState] = useState(() => {
    const traces: RoundingTrace[] = [];
    const ctrl = projectionController(sc.piLp, sc.model.budgets, (tr) => traces.push(tr));
    const plaza = new Plaza(sc.model, N, 'rationing', 21);
    for (let k = 0; k < 4; k++) plaza.step(ctrl);
    return { plaza, ctrl, traces, t: plaza.t };
  });
  const tr = state.traces[state.traces.length - 1];
  const t = (state.t + BASE.T - 1) % BASE.T;
  const next = () => {
    state.plaza.step(state.ctrl);
    setState({ ...state, t: state.plaza.t });
  };
  const rows = tr.counts.map((ns, s) => ({ s, ns })).filter((r) => r.ns > 0);
  const show = step >= 1 ? (step >= 4 ? tr.repaired : step >= 3 ? tr.rounded : null) : null;
  const totalA = (m: number[][]) => m.reduce((acc, r) => acc + r[2] + r[3], 0);
  const totalB = (m: number[][]) => m.reduce((acc, r) => acc + r[1] + r[3], 0);
  return (
    <div className="demo">
      <div className="rounding">
        <div className="rounding-head">
          Period <b>{t}</b>, {N} booths: <b>{capA}</b> cars and <b>{capB}</b> attendants available
        </div>
        <table>
          <thead>
            <tr>
              <th>cars <TeX>s</TeX></th>
              <th>booths <TeX>{'N_s'}</TeX></th>
              {[0, 1, 2, 3].map((u) => (
                <th key={u}><i className="swatch" style={{ background: PIE_COLORS[u] }} /> {ACTION_WORDS[u]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ s, ns }) => (
              <tr key={s}>
                <td>{s}</td>
                <td>{ns}</td>
                {[0, 1, 2, 3].map((u) => (
                  <td key={u}>
                    {step >= 2 && <span className="target">{tr.targets[s][u].toFixed(2)}</span>}
                    {show && <b className={show === tr.repaired && tr.repaired[s][u] !== tr.rounded[s][u] ? 'changed' : ''}> → {show[s][u]}</b>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {show && (
          <div className="rounding-foot">
            <span className={`pill ${totalA(show) <= capA ? 'ok' : 'bad'}`}>gates open: {totalA(show)} / {capA}</span>
            <span className={`pill ${totalB(show) <= capB ? 'ok' : 'bad'}`}>attendants: {totalB(show)} / {capB}</span>
            {step >= 4 && (tr.movesCars + tr.movesAtt > 0 ? <span className="muted">repaired {tr.movesCars + tr.movesAtt} booth(s)</span> : <span className="muted">no repair needed this time</span>)}
          </div>
        )}
      </div>
      <div className="demo-controls">
        <Button kind="primary" onClick={next}>Next period</Button>
      </div>
      {step >= 5 && (
        <BarChart
          title="The rounded plan, above the LP value (2000 simulated rush hours)"
          groups={['10 booths', '100 booths', '1000 booths']}
          series={[{ label: 'extra cost over the LP value (%)', color: COORD_COLOR, values: ['10', '100', '1000'].map((n) => (100 * (RESULTS.coordinators[0].costs[n] - RESULTS.lpValue)) / RESULTS.lpValue) }]}
          format={(v) => `${v.toFixed(1)}%`}
          height={180}
        />
      )}
    </div>
  );
}

export function SectionRounding() {
  return (
    <Scrolly
      id="rounding"
      number="11"
      title="From fractions to orders"
      kicker="Booths need a clear yes or no. The coordinator turns its plan into orders, every period."
      steps={[
        {
          key: 'a',
          body: (
            <>
              <p>The plan speaks in fractions. Dividing by the share of booths with <TeX>s</TeX> cars, <TeX>{'m_s(t) = \\sum_{a,b} y_{s,a,b}(t)'}</TeX>, gives the share of <i>those</i> booths that should do each action:</p>
              <TeX block>{'\\pi_t(a,b \\mid s) = \\frac{y_{s,a,b}(t)}{m_s(t)}'}</TeX>
              <p>For example: "one third of the booths with 3 cars open the gate". But a booth needs a clear yes or no. Here is what the coordinator does at each period.</p>
            </>
          ),
        },
        { key: 'b', body: <p><b>Count</b> how many booths have 0, 1, 2, … cars: <TeX>{'N_s'}</TeX>.</p> },
        { key: 'c', body: <p><b>Multiply</b> by the plan: <TeX>{'N_s\\, \\pi_t(a,b \\mid s)'}</TeX> booths should take action <TeX>{'(a,b)'}</TeX>. These targets are usually not whole numbers.</p> },
        { key: 'd', body: <p><b>Round</b> them at random, so that each row still adds up to <TeX>{'N_s'}</TeX> and the targets are kept on average.</p> },
        { key: 'e', body: <p><b>Repair</b>: if too many gates or attendants are assigned, take some back where the plan wanted them least. Then hand out the orders. The budgets now hold <b>every single period</b>. Press "Next period".</p> },
        {
          key: 'f',
          body: (
            <p>
              With few booths, the budgets are small whole numbers (10 booths get 7 cars and 2 attendants instead of 7.5 and 2.5) and rounding is clumsy. With many booths both effects fade, and the coordinator gets very close to the LP lower bound: {['10', '100', '1000'].map((n) => `${fmt((100 * (RESULTS.coordinators[0].costs[n] - RESULTS.lpValue)) / RESULTS.lpValue, 1)}%`).join(', ')} above it for 10, 100 and 1000 booths.
            </p>
          ),
        },
      ]}
      render={(step) => <RoundingDemo step={step} />}
    />
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// 12. Learning the plan

export function LearningPanel({ compact = false }: { compact?: boolean }) {
  const gda = useGda(BASE);
  const snap = gda.snap;
  const curve = useMemo(
    () =>
      gda.history
        .filter((h, i) => h.iter > 0 && (i % 2 === 0 || i === gda.history.length - 1))
        .map((h) => [h.iter, Math.min(480, fluidEvaluate(sc.model, conditionalPolicy(h.y), 'M1').cost)] as [number, number]),
    [gda.history],
  );
  const t = Array.from({ length: BASE.T }, (_, k) => k);
  const progress = snap ? snap.iter / gda.maxIter : 0;
  return (
    <div className={`learning ${compact ? 'compact' : ''}`}>
      <div className="demo-controls">
        {gda.running ? <Button onClick={gda.pause}>Pause learning</Button> : <Button kind="primary" onClick={gda.run} disabled={snap?.done}>{snap && snap.iter > 0 ? 'Keep learning' : 'Start learning'}</Button>}
        <Button kind="ghost" onClick={gda.reset}>Forget everything</Button>
        <div className="progress"><i style={{ width: `${100 * progress}%` }} /><span>{snap ? `${(snap.iter / 1000).toFixed(0)}k of ${gda.maxIter / 1000}k rounds` : '…'}</span></div>
      </div>
      <div className="learning-grid">
      <PiePlot y={snap && snap.iter > 0 ? snap.y : sc.lp.y.map((st) => st.map((r) => r.map(() => 0)))} title={snap && snap.iter > 0 ? 'The plan the AI has learned so far' : 'Nothing learned yet'} legend={!compact} compact={compact} />
      <div className="stacked-charts">
        <LineChart
          title="Prices: learned (solid) and solved (dashed)"
          xTicks={t}
          xLabel="period t"
          height={190}
          yDomain={[0, 80]}
          series={[
            ...(snap ? [
              { label: 'attendant, learned', color: COORD_COLOR, points: t.map((k) => [k, Math.min(80, snap.lam[1][k])] as [number, number]), marker: true },
              { label: 'car, learned', color: PRICE_COLOR, points: t.map((k) => [k, Math.min(80, snap.lam[0][k])] as [number, number]), marker: true },
            ] : []),
            { label: 'attendant, solved', color: '#7aa9cf', dash: '4 4', points: t.map((k) => [k, sc.lp.lam[1][k]] as [number, number]) },
            { label: 'car, solved', color: '#e8c27a', dash: '4 4', points: t.map((k) => [k, sc.lp.lam[0][k]] as [number, number]) },
          ]}
        />
        <LineChart
          title="Cost of the learned plan (many booths, extra requests by lottery)"
          xLabel="rounds of learning"
          height={190}
          xDomain={[0, gda.maxIter]}
          yDomain={[370, 480]}
          xTicks={[0, 300000, 600000, 900000, 1200000]}
          hLines={[
            { y: sc.lp.value, label: 'LP value (lower bound)', color: COORD_COLOR, below: true },
            { y: sc.fluid.selfishRationing.cost, label: 'selfish booths', color: SELFISH_COLOR },
          ]}
          series={[{ label: 'learned plan (values above 480 drawn at 480)', color: '#333', points: curve.length ? curve : [[0, 480]], marker: curve.length < 30 }]}
        />
      </div>
      </div>
    </div>
  );
}

export function SectionLearning() {
  return (
    <Scrolly
      id="learning"
      number="12"
      title="Computing the plan step by step"
      kicker="Instead of solving the LP in one go, the AI can compute its plan and its prices with many small adjustments."
      steps={[
        {
          key: 'a',
          body: (
            <p>
              Press <b>Start learning</b>. The AI starts with an empty plan and all prices at zero. Then, using the model's probabilities, it repeats two small adjustments about a million times. That takes 15 to 30 seconds in your browser.
            </p>
          ),
        },
        { key: 'b', body: <p><b>Price step:</b> if the plan uses more attendants than there are, raise the attendant's price; if it uses fewer, lower it (never below zero). Same for cars. It is like a market where prices rise when there is too much demand.</p> },
        { key: 'c', body: <p><b>Plan step:</b> shift the plan a little toward actions that are cheap at the current prices, while <i>trying</i> to respect the bookkeeping rules. This recipe is called <b>gradient descent–ascent</b> (GDA).</p> },
        {
          key: 'd',
          body: (
            <p>
              Watch the attendant's prices (solid) rise toward the solved ones (dashed), and the pies take the shape of the coordinator's plan from section 9: attendants on the longest lanes. Careful: the pies show every period as 100%. The plan itself never follows the rules exactly. By the last period it has "lost" about {fmt(100 * (1 - RESULTS.learning.gda.massFinal), 0)}% of the booths.
            </p>
          ),
        },
        {
          key: 'e',
          body: (
            <p>
              The right chart shows what the plan would cost with many booths if its extra requests were handed out by lottery. It falls from far above the selfish booths to close to the LP value. The plan also asks for more than exists, so when the coordinator uses it, it must correct about {fmt(100 * RESULTS.learning.gda.repairsPerPeriod, 0)}% of the booths every period, against {fmt(100 * RESULTS.learning.lp.repairsPerPeriod, 2)}% for the solved plan. Honest footnotes: this recipe uses the model's probabilities, so it does not learn from experience. A variant (SGDA) learns from simulated cars but needs about 480 million of them. And the car prices it finds do not match the solved ones.
            </p>
          ),
        },
      ]}
      render={() => <LearningPanel />}
      wide
    />
  );
}
