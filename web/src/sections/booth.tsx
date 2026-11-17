/** Part 1: one booth on its own. The state, time, the two decisions, what happens during a period, and the score. */
import { useEffect, useRef, useState } from 'react';
import { Scrolly } from '../components/Scrolly';
import { TeX } from '../components/TeX';
import { LaneView, useLane } from '../components/LaneView';
import { Button, Slider, Toggle } from '../components/controls';
import { BarChart } from '../components/charts';
import { LOST_COLOR, PIE_COLORS, ACTION_WORDS } from '../components/art';
import { BASE } from '../engine/data';
import { costParts, perPeriodDeparture, serviceProbability, transitionKernel } from '../engine/model';
import { Rng } from '../engine/rng';

const K = BASE.K;

// ---------------------------------------------------------------------------------------------------------------------
// 1. The booth and its state

function StateDemo({ step }: { step: number }) {
  const lane = useLane(2);
  const s = lane.queued;
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (step === 1 && s < K) lane.setCount(K);
  }, [step]); // eslint-disable-line react-hooks/exhaustive-deps
  const act = (fn: () => Promise<unknown>) => {
    setBusy(true);
    void fn().then(() => setBusy(false));
  };
  return (
    <div className="demo">
      <LaneView
        cars={lane.cars}
        gateOpen={lane.gateLifted}
        attendant={false}
        showSlots={step >= 1}
        full={s === K}
        caption={
          step >= 2 ? (
            <span className="state-readout">
              state <TeX>{`s = ${s}`}</TeX> {s === K ? ' (full lane!)' : s === 0 ? ' (empty lane)' : ''}
            </span>
          ) : (
            <span>{s} car{s === 1 ? '' : 's'} waiting{s === K ? ': the lane is full, a new car would be lost' : ''}</span>
          )
        }
      />
      <div className="demo-controls">
        <Button onClick={() => act(() => lane.arrive(true))} disabled={busy}>+ a car arrives</Button>
        <Button onClick={() => act(lane.payFirst)} disabled={busy || s === 0}>− the first car pays</Button>
      </div>
    </div>
  );
}

export function SectionState() {
  return (
    <Scrolly
      id="booth"
      number="1"
      title="Meet the booth"
      kicker="Everything starts with one toll booth and the line of cars in front of it."
      steps={[
        { key: 'a', body: <p>This is a <b>toll booth</b>. Cars wait in its <b>lane</b> until they have paid. Try the buttons: cars arrive, cars pay and leave.</p> },
        { key: 'b', body: <p>The lane is short: it holds at most <TeX>{`K = ${K}`}</TeX> cars. When it is full, a new car cannot get in and drives away without paying. That is lost money.</p> },
        {
          key: 'c',
          body: (
            <>
              <p>To decide what to do, the booth only needs one number: how many cars are waiting. We call it the <b>state</b>:</p>
              <TeX block>{`s \\in \\{0, 1, 2, 3, 4, 5\\}`}</TeX>
              <p>Six possible states, from an empty lane (<TeX>s=0</TeX>) to a full one (<TeX>s=5</TeX>).</p>
            </>
          ),
        },
      ]}
      render={(step) => <StateDemo step={step} />}
    />
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Time: steps, platoons and periods

function TimeDemo({ step, p, setP }: { step: number; p: number; setP: (p: number) => void }) {
  const [tape, setTape] = useState<boolean[]>([]);
  const [counts, setCounts] = useState<number[]>([]);
  const rng = useRef(new Rng(11));
  const timer = useRef<number | null>(null);
  const maxK = 8;
  useEffect(() => () => void (timer.current && clearTimeout(timer.current)), []);
  useEffect(() => {
    // A wait already in progress used the old probability. Discard it, along with the old samples.
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    setTape([]);
    setCounts([]);
  }, [p]);

  const record = (tau: number) => setCounts((c) => {
    const out = c.slice();
    while (out.length < maxK) out.push(0);
    out[Math.min(tau, maxK) - 1] += 1;
    return out;
  });
  const waitOne = () => {
    if (timer.current) clearTimeout(timer.current);
    setTape([]);
    const steps: boolean[] = [];
    const tick = () => {
      const arrives = rng.current.next() < p;
      steps.push(arrives);
      setTape(steps.slice());
      if (arrives) {
        record(steps.length);
        timer.current = null;
      } else timer.current = window.setTimeout(tick, 420);
    };
    timer.current = window.setTimeout(tick, 250);
  };
  const waitMany = () => {
    const out = counts.length ? counts.slice() : new Array(maxK).fill(0);
    for (let i = 0; i < 300; i++) out[Math.min(rng.current.geometric(p), maxK) - 1]++;
    setCounts(out);
  };
  const total = counts.reduce((a, b) => a + b, 0);
  const exact = Array.from({ length: maxK }, (_, k) => (k < maxK - 1 ? (1 - p) ** k * p : (1 - p) ** (maxK - 1)));
  return (
    <div className="demo">
      <div className="tape">
        {tape.length === 0 && <span className="muted">Press the button and watch the time steps tick.</span>}
        {tape.map((arrives, i) => (
          <div key={i} className={`tape-cell ${arrives ? 'hit' : ''}`}>
            <span>step {i + 1}</span>
            <b>{arrives ? 'platoon!' : 'no'}</b>
          </div>
        ))}
      </div>
      <div className="demo-controls">
        <Button kind="primary" onClick={waitOne}>Wait for the next platoon</Button>
        <Button onClick={waitMany}>Do it 300 times</Button>
        {step >= 1 && <Slider label="p =" value={p} min={0.1} max={0.95} step={0.05} onChange={setP} format={(v) => v.toFixed(2)} />}
      </div>
      {step >= 2 && (
        <BarChart
          title={<>How many steps until the platoon? ({total} tries)</>}
          yLabel="share of tries"
          groups={Array.from({ length: maxK }, (_, k) => (k === maxK - 1 ? `${maxK}+` : String(k + 1)))}
          series={[
            { label: 'your tries', color: '#669bbc', values: counts.length ? counts.map((c) => c / Math.max(total, 1)) : new Array(maxK).fill(0) },
            { label: 'formula', color: '#29335c', values: exact, outline: true },
          ]}
          yMax={1}
          height={190}
          note="8+ means eight or more steps. Changing p clears the tries."
        />
      )}
    </div>
  );
}

export function SectionTime() {
  const [p, setP] = useState(BASE.p);
  return (
    <Scrolly
      id="time"
      number="2"
      title="How time passes"
      kicker="Cars come in groups, and nobody knows exactly when the next group arrives."
      steps={[
        { key: 'a', body: <p>Time is made of small <b>time steps</b>. Cars do not arrive one by one: a whole <b>platoon</b> of cars arrives together, for instance when a traffic light up the road turns green.</p> },
        {
          key: 'b',
          body: (
            <p>
              In each time step the next platoon arrives with probability <TeX>{`p = ${p.toFixed(2)}`}</TeX>, like a biased coin. Press the button a few times. Then change <TeX>p</TeX>: a small <TeX>p</TeX> means long waits.
            </p>
          ),
        },
        {
          key: 'c',
          body: (
            <>
              <p>The number of steps <TeX>\tau</TeX> until the platoon arrives follows a <b>geometric</b> law:</p>
              <TeX block>{`\\Pr\\{\\tau = k\\} = (1-p)^{k-1}\\,p`}</TeX>
              <p>On average you wait <TeX>{'\\mathbb{E}[\\tau] = 1/p'}</TeX> steps: about {(1 / p).toFixed(2)} steps for <TeX>{`p = ${p.toFixed(2)}`}</TeX>.</p>
              <p>First <TeX>{'k-1'}</TeX> "no", then one "yes". Run it 300 times and the bars should be close to the formula; luck makes them a little different.</p>
            </>
          ),
        },
        {
          key: 'd',
          body: (
            <p>
              The time between two platoons is a <b>period</b>. The booth decides only when a platoon arrives, at the start of each period. We follow one <b>rush hour</b> of <TeX>{`T = ${BASE.T}`}</TeX> periods, numbered <TeX>{`t = 0, 1, \\dots, ${BASE.T - 1}`}</TeX>, starting with an empty lane.
            </p>
          ),
        },
      ]}
      render={(step) => <TimeDemo step={step} p={p} setP={setP} />}
    />
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Two decisions

function ActionsDemo({ step }: { step: number }) {
  const [a, setA] = useState(1);
  const [b, setB] = useState(0);
  useEffect(() => {
    if (step === 1) setA(1);
    if (step === 2) setB(1);
  }, [step]);
  const lane = useLane(2);
  const u = 2 * a + b;
  return (
    <div className="demo">
      <LaneView cars={lane.cars} gateOpen={a === 1} attendant={b === 1} />
      <div className="demo-controls">
        <Toggle label={<>Gate <TeX>a</TeX></>} value={a} onChange={setA} options={[{ value: 0, label: 'closed (0)' }, { value: 1, label: 'open (1)' }]} />
        <Toggle label={<>Attendant <TeX>b</TeX></>} value={b} onChange={setB} options={[{ value: 0, label: 'no (0)' }, { value: 1, label: 'yes (1)' }]} />
      </div>
      {step >= 3 && (
        <div className="action-cards">
          {[0, 1, 2, 3].map((k) => (
            <button key={k} type="button" className={`action-card ${k === u ? 'on' : ''}`} onClick={() => { setA(k >> 1); setB(k & 1); }}>
              <i style={{ background: PIE_COLORS[k] }} />
              <TeX>{`(a,b) = (${k >> 1},${k & 1})`}</TeX>
              <span>{ACTION_WORDS[k]}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function SectionActions() {
  return (
    <Scrolly
      id="actions"
      number="3"
      title="Two buttons"
      kicker="At the start of every period, each booth answers two yes-or-no questions."
      steps={[
        { key: 'a', body: <p>When a platoon arrives, the booth makes two decisions. Each one is a 0 (no) or a 1 (yes).</p> },
        { key: 'b', body: <p><b>Open the gate?</b> <TeX>{'a \\in \\{0, 1\\}'}</TeX>. With <TeX>a = 1</TeX> the booth lets one car of the platoon into its lane. Each car pays a toll, so the booth wants cars.</p> },
        { key: 'c', body: <p><b>Call a mobile attendant?</b> <TeX>{'b \\in \\{0, 1\\}'}</TeX>. An attendant walks along the line with a card machine and makes every car pay <b>ten times faster</b>. But attendants cost money too.</p> },
        {
          key: 'd',
          body: (
            <>
              <p>Together, the two answers are the booth's <b>action</b>:</p>
              <TeX block>{'u = (a, b) \\in \\{(0,0), (0,1), (1,0), (1,1)\\}'}</TeX>
              <p>Remember the four colours. Later, they show what thousands of booths are doing at once.</p>
            </>
          ),
        },
      ]}
      render={(step) => <ActionsDemo step={step} />}
    />
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. What happens during a period

function TransitionDemo({ step }: { step: number }) {
  const [s, setS] = useState(3);
  const [a, setA] = useState(1);
  const [b, setB] = useState(0);
  const lane = useLane(3);
  const [status, setStatus] = useState('');
  const [hist, setHist] = useState<number[]>(() => new Array(K + 1).fill(0));
  const [running, setRunning] = useState(false);
  const rng = useRef(new Rng(5));
  useEffect(() => {
    lane.setCount(s);
    setStatus('');
  }, [s]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => setHist(new Array(K + 1).fill(0)), [s, a, b]);
  const q = serviceProbability(b, BASE);
  const exact = transitionKernel(K, BASE.p, q)[a][s];

  const playOne = async () => {
    if (running) return;
    setRunning(true);
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
    if (lane.waiting().length !== s) {
      lane.reset(s);
      await sleep(80);
    }
    setStatus('A platoon arrives…');
    const result = await lane.arrive(a === 1);
    setStatus(
      result === 'entered'
        ? 'The gate is open and there is room: one car gets in.'
        : a === 1
          ? 'The lane is full: the car cannot get in and is lost.'
          : 'The gate is closed: the car goes elsewhere (a missed toll).',
    );
    await sleep(500);
    const tau = rng.current.geometric(BASE.p);
    for (let k = 1; k <= tau; k++) {
      const paying = lane.waiting().filter(() => rng.current.next() < q).map((c) => c.id);
      setStatus(`Time step ${k} of ${tau}: each car pays with probability ${q}.${paying.length ? ` ${paying.length} car${paying.length > 1 ? 's' : ''} paid!` : ''}`);
      await Promise.all([lane.pay(paying), sleep(700)]);
    }
    const left = lane.waiting().length;
    setStatus(`The next platoon arrives: now s' = ${left}.`);
    setHist((h) => h.map((v, i) => (i === left ? v + 1 : v)));
    setRunning(false);
  };
  const playMany = () => {
    const out = hist.slice();
    for (let i = 0; i < 500; i++) {
      const n = s < K ? s + a : s;
      const tau = rng.current.geometric(BASE.p);
      let stay = 0;
      for (let c = 0; c < n; c++) if (rng.current.geometric(q) > tau) stay++;
      out[stay]++;
    }
    setHist(out);
  };
  const total = hist.reduce((x, y) => x + y, 0);
  return (
    <div className="demo">
      <LaneView cars={lane.cars} gateOpen={a === 1} attendant={b === 1} caption={status || 'Pick a situation and play a period.'} />
      <fieldset className="demo-controls" disabled={running}>
        <Slider label={<>cars now, <TeX>s</TeX> =</>} value={s} min={0} max={K} onChange={setS} />
        <Toggle label="gate" value={a} onChange={setA} options={[{ value: 0, label: 'closed' }, { value: 1, label: 'open' }]} />
        <Toggle label="attendant" value={b} onChange={setB} options={[{ value: 0, label: 'no' }, { value: 1, label: 'yes' }]} />
      </fieldset>
      <fieldset className="demo-controls" disabled={running}>
        <Button kind="primary" onClick={playOne}>Play one period</Button>
        {step >= 3 && <Button onClick={playMany}>Play 500 periods</Button>}
      </fieldset>
      {step >= 3 && (
        <BarChart
          title={<>Cars at the next period, <TeX>s'</TeX> ({total} periods played)</>}
          groups={Array.from({ length: K + 1 }, (_, k) => String(k))}
          series={[
            { label: 'played', color: '#669bbc', values: hist.map((v) => (total ? v / total : 0)) },
            { label: "P(s' | s, a, b), the formula", color: '#29335c', values: exact, outline: true },
          ]}
          yMax={1}
          height={190}
        />
      )}
    </div>
  );
}

export function SectionTransitions() {
  const r0 = perPeriodDeparture(BASE.p, BASE.bLow);
  const r1 = perPeriodDeparture(BASE.p, BASE.bHigh);
  return (
    <Scrolly
      id="transitions"
      number="4"
      title="What happens during a period"
      kicker="Cars enter, cars pay, and luck decides how many are left when the next platoon arrives."
      steps={[
        {
          key: 'a',
          body: (
            <>
              <p><b>A car may enter.</b> If the gate is open and there is room, one car joins the lane. If the lane is full, the car is lost. With <TeX>{'\\mathbf 1\\{\\cdot\\}'}</TeX> meaning "1 if true, 0 if not":</p>
              <TeX block>{'s^+ = s + a \\cdot \\mathbf{1}\\{s < K\\}'}</TeX>
            </>
          ),
        },
        {
          key: 'b',
          body: (
            <>
              <p><b>Cars pay and leave.</b> In each time step, each car finishes paying with a small probability:</p>
              <TeX block>{`q(0) = ${BASE.bLow} \\text{ (normal)}, \\qquad q(1) = ${BASE.bHigh} \\text{ (with an attendant)}`}</TeX>
              <p>Press "Play one period" and watch each car flip its own coin at every step until the next platoon.</p>
            </>
          ),
        },
        {
          key: 'c',
          body: (
            <p>
              Over a whole period, a car leaves with probability about <b>{(100 * r0).toFixed(1)}%</b> at normal speed and <b>{(100 * r1).toFixed(1)}%</b> with an attendant. Because every car in the lane gets faster, an attendant helps a long lane more than a short one. Keep this in mind!
            </p>
          ),
        },
        {
          key: 'd',
          body: (
            <>
              <p>Luck decides where we end up. Call <TeX>{"\\rho_\\tau = (1-q(b))^{\\tau}"}</TeX> the chance that a car is still in the lane after <TeX>{"\\tau"}</TeX> steps. The chance of ending with <TeX>{"s'"}</TeX> cars is the <b>transition probability</b>:</p>
              <TeX block>{"P(s' \\mid s, a, b) = \\sum_{\\tau \\ge 1} \\underbrace{(1-p)^{\\tau-1} p}_{\\tau \\text{ steps}} \\binom{s^+}{s'} \\rho_\\tau^{\\,s'} (1-\\rho_\\tau)^{s^+ - s'}"}</TeX>
              <p>Don't panic: play 500 periods and the bars land on this formula.</p>
            </>
          ),
        },
        {
          key: 'e',
          body: (
            <p>
              Notice what the formula needs: only the state now and the action now. Not the past. That is the <b>Markov property</b>, the "M" in MDP.
            </p>
          ),
        },
      ]}
      render={(step) => <TransitionDemo step={step} />}
    />
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. Keeping score

function CostDemo({ step }: { step: number }) {
  const [s, setS] = useState(2);
  const [a, setA] = useState(1);
  const [b, setB] = useState(1);
  const parts = costParts(BASE, s, a, b);
  const lane = useLane(2);
  useEffect(() => lane.setCount(s), [s]); // eslint-disable-line react-hooks/exhaustive-deps
  const rows = [
    { key: 'waiting', label: 'waiting', value: parts.waiting, color: '#669bbc', show: step >= 1, tex: `s/K = ${s}/${K}` },
    { key: 'staff', label: 'staff', value: parts.staff, color: '#f3a712', show: step >= 2, tex: `2(1+q(${b}))` },
    { key: 'missed', label: 'missed toll', value: parts.missed, color: LOST_COLOR, show: step >= 3, tex: parts.missed > 0 ? '\\gamma = 100' : '0' },
  ];
  const shown = rows.filter((r) => r.show);
  const total = shown.reduce((x, r) => x + r.value, 0);
  const scale = (v: number) => `${Math.min(100, (v / 105) * 100)}%`;
  return (
    <div className="demo">
      <LaneView cars={lane.cars} gateOpen={a === 1} attendant={b === 1} />
      <div className="demo-controls">
        <Slider label={<>cars before arrival, <TeX>s</TeX> =</>} value={s} min={0} max={K} onChange={setS} />
        <Toggle label="gate" value={a} onChange={setA} options={[{ value: 0, label: 'closed' }, { value: 1, label: 'open' }]} />
        <Toggle label="attendant" value={b} onChange={setB} options={[{ value: 0, label: 'no' }, { value: 1, label: 'yes' }]} />
      </div>
      <div className="receipt">
        <div className="receipt-title">This period's bill</div>
        {shown.map((r) => (
          <div key={r.key} className="receipt-row">
            <span className="receipt-label">{r.label} <TeX>{r.tex}</TeX></span>
            <span className="receipt-bar"><i style={{ width: scale(r.value), background: r.color }} /></span>
            <b>{r.value.toFixed(2)}</b>
          </div>
        ))}
        {step >= 4 && (
          <div className="receipt-row total">
            <span className="receipt-label">total <TeX>{'c(s,a,b)'}</TeX></span>
            <span />
            <b>{total.toFixed(2)}</b>
          </div>
        )}
        {step >= 3 && (
          <div className={`receipt-note${parts.missed === 0 ? ' success' : ''}`}>
            {a === 0
              ? 'The gate is closed: no car enters, so the toll is missed.'
              : s === K
                ? 'The gate is open, but the lane is full: no car enters, so the toll is missed.'
                : 'The gate is open and there is room: one car enters, so no toll is missed.'}
          </div>
        )}
      </div>
    </div>
  );
}

export function SectionCosts() {
  return (
    <Scrolly
      id="costs"
      number="5"
      title="Keeping score"
      kicker="Every period has a price. The booth wants the total over the rush hour to be as small as possible."
      steps={[
        { key: 'a', body: <p>We keep score with a <b>cost</b>, counted in simple "cost units": lower is better. (Some books use a <i>reward</i> instead; a reward is just a cost with the sign flipped.) Each period, the booth pays three kinds of cost.</p> },
        { key: 'b', body: <p><b>Waiting:</b> drivers hate waiting. Each period costs <TeX>{'s/K'}</TeX>, from 0 for an empty lane to 1 for a full one.</p> },
        { key: 'c', body: <p><b>Staff:</b> running the booth costs <TeX>{'2(1+q(b))'}</TeX>: 2.02 at normal speed and 2.20 with an attendant. An attendant costs only <b>0.18</b> extra.</p> },
        { key: 'd', body: <p><b>Missed toll:</b> at the start of the period, if no new car enters because the gate is closed or the lane is full, the booth pays <TeX>{'\\gamma = 100'}</TeX>. A car that enters avoids this charge even if it leaves later. That is huge compared with the rest: cars are precious.</p> },
        {
          key: 'e',
          body: (
            <>
              <p>All together, the cost of one period is:</p>
              <TeX block>{'c(s,a,b) = \\frac{s}{K} + 2\\big(1+q(b)\\big) + \\gamma\\,\\big(1 - a\\,\\mathbf 1\\{s<K\\}\\big)'}</TeX>
              <p>Over the rush hour the booth wants the smallest <i>expected total</i> (capital letters are the random number of cars and the actions at period <TeX>t</TeX>):</p>
              <TeX block>{'\\min\\; \\mathbb{E}\\Big[\\sum_{t=0}^{T-1} c\\big(S(t), A(t), B(t)\\big)\\Big]'}</TeX>
              <p>States, actions, transition probabilities, costs: we now have a complete <b>Markov decision process</b> (MDP).</p>
            </>
          ),
        },
      ]}
      render={(step) => <CostDemo step={step} />}
    />
  );
}
