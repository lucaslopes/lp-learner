/** Part 3: many booths share the platoon and the attendants; then everybody plays selfishly. */
import { useMemo, useState } from 'react';
import { Scrolly } from '../components/Scrolly';
import { TeX } from '../components/TeX';
import { PlazaStage } from '../components/PlazaStage';
import { Button, Slider, Stat, Toggle } from '../components/controls';
import { BarChart } from '../components/charts';
import { BASE, RESULTS, scenario, fmt } from '../engine/data';
import { capacities } from '../engine/model';
import { tableController, type PeriodOutcome, type Rule } from '../engine/sim';

const sc = scenario();

function ManyBoothsDemo({ step }: { step: number }) {
  const [N, setN] = useState(8);
  const [bigN, setBigN] = useState(20);
  const [capA, capB] = capacities(N, BASE.alpha, BASE.beta);
  const idle = useMemo(() => tableController('idle', Array.from({ length: BASE.T }, () => new Array(BASE.K + 1).fill(0))), []);
  const exponent = bigN * Math.log10(BASE.K + 1);
  return (
    <div className="demo">
      {step < 4 ? (
        <>
          <PlazaStage model={sc.model} N={N} rule="rationing" controller={idle} playing={false} speed={1} showRequests={false} height={330} />
          <div className="demo-controls">
            <Slider label="booths N =" value={N} min={2} max={16} onChange={setN} />
            <span className="pill">platoon: <b>{capA}</b> cars</span>
            <span className="pill">attendants: <b>{capB}</b></span>
          </div>
        </>
      ) : (
        <div className="explosion">
          <Slider label="booths N =" value={bigN} min={2} max={1000} onChange={setBigN} />
          <div className="big-number">
            <TeX>{`6^{${bigN}} \\approx 10^{${Math.floor(exponent)}}`}</TeX>
          </div>
          <div className="muted">possible states of the whole plaza (each booth has 6 states)</div>
          <div className="compare-bar">
            <div style={{ width: `${Math.min(100, (exponent / 80) * 100)}%` }} />
          </div>
          <div className="muted">{exponent < 80 ? `the bar fills up at 10^80, roughly the number of atoms in the observable universe` : `more than the number of atoms in the observable universe (about 10^80)!`}</div>
        </div>
      )}
    </div>
  );
}

export function SectionManyBooths() {
  return (
    <Scrolly
      id="many"
      number="7"
      title="A thousand booths, a few helpers"
      kicker="Booths do not live alone. They share the cars of each platoon and the plaza's few attendants."
      steps={[
        { key: 'a', body: <p>Now put <TeX>N</TeX> booths side by side. Each has its own lane, its own state and its own decisions. Move the slider.</p> },
        { key: 'b', body: <p>But the platoon has fewer cars than there are booths: <TeX>{`\\alpha N`}</TeX> cars with <TeX>{`\\alpha = ${BASE.alpha}`}</TeX>, so three cars for every four booths. Each booth can take at most one. So in every period at least a quarter of the booths get no car and miss a toll, whatever anybody does.</p> },
        { key: 'c', body: <p>And the plaza has only <TeX>{`\\beta N`}</TeX> mobile attendants, with <TeX>{`\\beta = ${BASE.beta}`}</TeX>: one for every four booths.</p> },
        {
          key: 'd',
          body: (
            <>
              <p>In every period, the decisions of all booths must fit the budgets:</p>
              <TeX block>{'\\sum_{n=1}^{N} a_n \\le \\alpha N, \\qquad \\sum_{n=1}^{N} b_n \\le \\beta N'}</TeX>
              <p>The booths are independent MDPs, linked only by these two sums. This is a <b>weakly coupled MDP</b>.</p>
            </>
          ),
        },
        {
          key: 'e',
          body: (
            <p>
              Why not just solve the whole plaza with Bellman's equation? Its state is the list of all lane lengths, and there are <TeX>{'6^N'}</TeX> such lists. Slide to 1000 booths.
            </p>
          ),
        },
      ]}
      render={(step) => <ManyBoothsDemo step={step} />}
    />
  );
}

function SelfishDemo({ step }: { step: number }) {
  const [rule, setRule] = useState<Rule>('rationing');
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [last, setLast] = useState<PeriodOutcome | null>(null);
  const [hours, setHours] = useState<number[]>([]);
  const ctrl = useMemo(() => tableController('selfish', sc.selfish.u), []);
  const N = 12;
  return (
    <div className="demo">
      <PlazaStage
        model={sc.model} N={N} rule={rule} controller={ctrl} playing={playing && step >= 0} speed={speed} seed={7}
        onPeriod={setLast} onRushHour={(c) => setHours((h) => [...h.slice(-11), c])} height={360}
      />
      <div className="demo-controls">
        <Button onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</Button>
        <Toggle label="speed" value={speed} onChange={setSpeed} options={[{ value: 0.5, label: '½×' }, { value: 1, label: '1×' }, { value: 3, label: '3×' }]} />
        {step >= 3 && (
          <Toggle label="too many requests?" value={rule} onChange={(r) => { setRule(r); setHours([]); }}
            options={[{ value: 'rationing', label: 'lottery' }, { value: 'congestion', label: 'share attendants' }]} />
        )}
      </div>
      {last && step >= 1 && (
        <BarChart
          title="This period: requests and supply"
          groups={['cars', 'attendants']}
          series={[
            { label: 'asked for', color: '#009E73', values: [last.requestsA, last.requestsB] },
            { label: 'available', color: '#555', values: [last.capA, last.capB], outline: true },
          ]}
          format={(v) => v.toFixed(0)}
          height={170}
          yMax={N + 1}
        />
      )}
      {step >= 4 && (
        <div className="stats-row">
          <Stat tone="selfish" label="attendants asked for, per attendant" value={fmt(RESULTS.detail.selfishRationing.attRequestsPerAtt, 1) + '×'} sub="lottery, 1000 booths, 2000 simulated rush hours" />
          <Stat tone="selfish" label="periods with too many requests" value={`${fmt(100 * RESULTS.detail.selfishRationing.overloadAtt, 0)}%`} sub="for the attendants (lottery, 1000 booths)" />
          <Stat label="cost per booth, this demo" value={hours.length ? fmt(hours[hours.length - 1], 1) : '…'} sub={`last rush hour, ${N} booths`} />
        </div>
      )}
    </div>
  );
}

export function SectionSelfish() {
  return (
    <Scrolly
      id="selfish"
      number="8"
      title="Everybody for themselves"
      kicker="What happens if every booth follows its own selfish plan?"
      steps={[
        { key: 'a', body: <p>Here, every booth follows the <b>selfish policy</b> from the table we just built. It is the best plan for one booth on its own.</p> },
        { key: 'b', body: <p>Look at the speech bubbles: almost every booth asks for a car <i>and</i> an attendant. Early in the rush hour, all of them do. The bars compare what is asked for with what exists.</p> },
        {
          key: 'c',
          body: (
            <p>
              There are not enough cars and attendants for everyone. The plaza cannot hand out what it does not have, so a <b>lottery</b> decides: winners get one, the others don't. A booth that asks and loses is no worse off than if it had not asked, so asking never hurts. <b>Everybody asks.</b>
            </p>
          ),
        },
        { key: 'd', body: <p>Another possibility: the attendants <b>share</b> their time among all the booths that called them. Each caller pays for a full attendant but gets only a slice of one. Try both rules.</p> },
        {
          key: 'e',
          body: (
            <p>
              In our big simulations (1000 booths, 2000 rush hours, lottery), the selfish booths ask for <b>{fmt(RESULTS.detail.selfishRationing.attRequestsPerAtt, 1)} times</b> as many attendants as exist, and there are too many requests in <b>{fmt(100 * RESULTS.detail.selfishRationing.overloadAtt, 0)}%</b> of the periods. Is that bad? We need something to compare with: a plaza with a coordinator.
            </p>
          ),
        },
      ]}
      render={(step) => <SelfishDemo step={step} />}
    />
  );
}
