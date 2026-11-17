/** Part 5: the verdict, from the recorded simulations (1000 booths, 2000 rush hours per setting). */
import { Scrolly } from '../components/Scrolly';
import { BarChart, LineChart } from '../components/charts';
import { COORD_COLOR, CONGESTION_COLOR, SELFISH_COLOR } from '../components/art';
import { RESULTS, fmt, signed } from '../engine/data';

function VerdictDemo({ step }: { step: number }) {
  const m = RESULTS.main;
  if (step === 0)
    return (
      <BarChart
        title="Extra cost of the selfish booths, compared with the coordinator"
        groups={m.map((r) => `${r.N} booths`)}
        series={[
          { label: 'lottery (rationing)', color: SELFISH_COLOR, values: m.map((r) => r.rationing.pct) },
          { label: 'shared attendants (congestion)', color: CONGESTION_COLOR, values: m.map((r) => r.congestion.pct) },
        ]}
        yLabel="%"
        format={(v) => `+${v.toFixed(1)}%`}
        note={`Cost per booth over a rush hour, ${RESULTS.replications} simulated rush hours per setting, same random numbers for both.`}
      />
    );
  if (step === 1) {
    const t = Array.from({ length: RESULTS.fluidLostCumulative.lp.length }, (_, k) => k);
    return (
      <LineChart
        title="Cars lost per booth, adding up over the rush hour (many booths)"
        xLabel="period t"
        xTicks={t}
        series={[
          { label: 'coordinator', color: COORD_COLOR, marker: true, points: t.map((k) => [k, RESULTS.fluidLostCumulative.lp[k]]) },
          { label: 'selfish, lottery', color: SELFISH_COLOR, marker: true, points: t.map((k) => [k, RESULTS.fluidLostCumulative.selfish_M1[k]]) },
          { label: 'selfish, shared attendants', color: CONGESTION_COLOR, marker: true, points: t.map((k) => [k, RESULTS.fluidLostCumulative.selfish_M2[k]]) },
        ]}
      />
    );
  }
  if (step === 2) {
    const d = RESULTS.dilemma.rationing;
    const cards = [
      { title: 'Everybody selfish instead of coordinated', value: d.collective, who: 'every booth, on average', bad: true },
      { title: 'One booth stops calling attendants', value: d.deviations.neverAttendant, who: 'that booth', bad: true },
      { title: "One booth follows the coordinator's plan alone", value: d.deviations.followCoordinator, who: 'that booth', bad: true },
      { title: 'One booth stops opening its gate', value: d.deviations.neverGate, who: 'that booth', bad: true },
    ];
    return (
      <div className="cards">
        {cards.map((c, i) => (
          <div key={c.title} className={`card ${i === 0 ? 'card-collective' : 'card-individual'}`}>
            <div className="card-title">{c.title}</div>
            <div className="card-value">{signed(c.value.diff, c.value.ci < 1 ? 1 : 0)} <small>± {fmt(c.value.ci, c.value.ci < 1 ? 1 : 0)}</small></div>
            <div className="card-sub">change of the cost of {c.who} over a rush hour (lottery, 1000 booths; ± = 95% range)</div>
          </div>
        ))}
      </div>
    );
  }
  if (step === 3)
    return (
      <BarChart
        title="What if one resource were plentiful? (1000 booths)"
        groups={RESULTS.whatif.map((w) => w.label)}
        series={[
          { label: 'lottery', color: SELFISH_COLOR, values: RESULTS.whatif.map((w) => w.rationing) },
          { label: 'shared attendants', color: CONGESTION_COLOR, values: RESULTS.whatif.map((w) => w.congestion) },
        ]}
        yLabel="%"
        format={(v) => `${signed(v)}%`}
        note="Extra cost of the selfish booths over the coordinator. Below zero, the selfish booths are cheaper."
      />
    );
  return (
    <div className="limits">
      <h3>What this does and does not show</h3>
      <ul>
        <li>The toll plaza is a <b>picture</b>. The model was built for computer servers sharing jobs and power; real toll plazas work differently.</li>
        <li>One small example: lanes of 5 cars, 10 periods, every car can pay at the same time, and each lane has its own clock.</li>
        <li>What happens when too many booths ask (lottery, sharing) is our assumption. If the plaza could magically create attendants, the selfish booths would look cheaper, because they would use helpers that do not exist.</li>
        <li>Our coordinator is very good with many booths, but it is not perfect. The LP value is a bound, not a policy.</li>
        <li>We did not model booths that talk to each other, learn, or meet every day. That is often how real communities escape the tragedy.</li>
      </ul>
    </div>
  );
}

export function SectionVerdict() {
  const r = RESULTS.main[RESULTS.main.length - 1];
  const d = RESULTS.detail;
  const dil = RESULTS.dilemma.rationing.deviations;
  const lostMore = (100 * (d.selfishRationing.lostCars - d.coord.lostCars)) / d.coord.lostCars;
  const wiAtt = RESULTS.whatif.find((w) => w.scenario === 'beta=1.00');
  const wiCars = RESULTS.whatif.find((w) => w.scenario === 'alpha=1.00');
  const sens = (scenario: string) => fmt(RESULTS.sensitivity.find((x) => x.scenario === scenario)?.rationingPct ?? NaN, 1);
  return (
    <Scrolly
      id="verdict"
      number="13"
      title="The verdict: a tragedy of the commons"
      kicker={`Results from our big simulations (1000 booths, ${RESULTS.replications} rush hours for each setting, the same luck for every policy), and, where marked, the many-booth limit computed exactly. "±" is the 95% uncertainty range of the simulations.`}
      steps={[
        {
          key: 'a',
          body: (
            <p>
              With 1000 booths, the coordinator costs <b>{fmt(r.coord.mean, 1)}</b> per booth over a rush hour. The selfish booths cost <b>{fmt(r.rationing.mean, 1)}</b> with the lottery, <b>{signed(r.rationing.pct)}%</b>, and <b>{fmt(r.congestion.mean, 1)}</b> when attendants are shared, <b>{signed(r.congestion.pct)}%</b>. The more booths, the bigger the gap. A few percent may sound small, but {fmt(RESULTS.unavoidableMissed, 0)} of each booth's cost cannot be avoided by anyone: there are only 3 cars for every 4 booths, so at least a quarter of the booths miss a toll in every period.
            </p>
          ),
        },
        {
          key: 'b',
          body: (
            <p>
              Almost all of the difference is <b>lost cars</b>. Of the {fmt(RESULTS.carsPerBooth, 1)} cars that arrive per booth during a rush hour, the coordinator loses {fmt(d.coord.lostCars, 2)} and the selfish booths lose {fmt(d.selfishRationing.lostCars, 2)}, {fmt(lostMore, 0)}% more. Attendants spread by lottery let lanes fill up sooner, and full lanes lose cars. The chart shows the many-booth limit: there, the coordinator loses no car before period 7.
            </p>
          ),
        },
        {
          key: 'c',
          body: (
            <>
              <p>
                Is it really a <b>tragedy of the commons</b>? Two tests. <b>Together</b>, being selfish is worse for everybody. <b>Alone</b>, can one booth do better by changing its behaviour while the others stay selfish? No. Every change we tried makes that booth pay more over a rush hour: {signed(dil.neverAttendant.diff, 0)} if it never calls an attendant, {signed(dil.followCoordinator.diff, 0)} if it follows the coordinator's plan on its own, {signed(dil.neverGate.diff, 0)} if it never opens its gate. In the many-booth limit, we also checked every possible behaviour of one booth: none beats being selfish.
              </p>
              <p>So each booth does the best it can, and everybody ends up worse off. That is exactly the trap Garrett Hardin described in 1968 for herders sharing a pasture.</p>
            </>
          ),
        },
        {
          key: 'd',
          body: (
            <p>
              The tragedy depends on what is scarce. With an attendant for every booth it disappears ({wiAtt ? signed(wiAtt.rationing) : ''}%: the selfish booths are even a bit cheaper, partly because our coordinator is itself {wiAtt ? fmt((100 * (wiAtt.coord.mean - wiAtt.lpValue)) / wiAtt.lpValue, 1) : ''}% above its lower bound there). With a car for every booth it almost disappears ({wiCars ? signed(wiCars.rationing) : ''}% with the lottery). And it is not simply "scarcer is worse": with 100 booths, the loss is {sens('beta=0.10')}% with fewer attendants (β = 0.10), {sens('base')}% in the base case and {sens('beta=0.50')}% with more (β = 0.50). The loss is real but modest: a few percent, not a collapse. Try other budgets in the playground.
            </p>
          ),
        },
        { key: 'e', body: <p>And now: be honest about what we did not show. Then go and play.</p> },
      ]}
      render={(step) => <VerdictDemo step={step} />}
    />
  );
}
