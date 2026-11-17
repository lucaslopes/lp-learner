/** Part 2: one booth plans its rush hour backwards (backward induction / Bellman's recursion). */
import { useEffect, useState } from 'react';
import { Scrolly } from '../components/Scrolly';
import { TeX } from '../components/TeX';
import { LineChart } from '../components/charts';
import { ACTION_WORDS, PIE_COLORS } from '../components/art';
import { BASE, scenario } from '../engine/data';
import { attendantGain } from '../engine/dp';
import { ADMIT, HIGH } from '../engine/model';

const sc = scenario();

function BellmanTable({ step }: { step: number }) {
  const { T, K } = BASE;
  const sol = sc.selfish;
  const [revealFrom, setRevealFrom] = useState(T);
  const [hover, setHover] = useState<{ t: number; s: number } | null>(null);
  useEffect(() => {
    if (step === 0) setRevealFrom(T);
    else if (step === 1) setRevealFrom(T - 1);
    else if (step === 2) {
      setRevealFrom(T - 1);
      let col = T - 1;
      const id = window.setInterval(() => {
        col -= 1;
        setRevealFrom(col);
        if (col <= 0) window.clearInterval(id);
      }, 420);
      return () => window.clearInterval(id);
    } else setRevealFrom(0);
    return undefined;
  }, [step, T]);
  const shown = (t: number) => t >= revealFrom;
  const cellW = 52;
  const cellH = 46;
  const padL = 46;
  const padT = 26;
  const width = padL + T * cellW + 6;
  const height = padT + (K + 1) * cellH + 34;
  const h = hover ?? { t: Math.max(revealFrom, 0) < T ? Math.max(revealFrom, 0) : T - 1, s: 0 };
  const q = sol.Q[h.t]?.[h.s];
  return (
    <div className="demo">
      <svg className="bellman" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Backward induction table">
        <text x={padL} y={16} className="chart-title">{'V'}<tspan baselineShift="sub" fontSize="10">t</tspan>(s): best expected cost from period t to the end</text>
        {Array.from({ length: T }, (_, t) => (
          <text key={t} x={padL + (t + 0.5) * cellW} y={height - 18} textAnchor="middle" className="axis-tick">{t}</text>
        ))}
        <text x={padL + (T * cellW) / 2} y={height - 2} textAnchor="middle" className="axis-label">period t</text>
        {Array.from({ length: K + 1 }, (_, s) => (
          <text key={s} x={padL - 10} y={padT + (K - s + 0.5) * cellH + 4} textAnchor="end" className="axis-tick">{s}</text>
        ))}
        <text transform={`translate(12, ${padT + ((K + 1) * cellH) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">cars s</text>
        {Array.from({ length: T }, (_, t) =>
          Array.from({ length: K + 1 }, (_, s) => {
            const x = padL + t * cellW;
            const y = padT + (K - s) * cellH;
            const u = sol.u[t][s];
            const on = shown(t);
            const reachable = s <= t;
            return (
              <g key={`${t}-${s}`} onMouseEnter={() => on && setHover({ t, s })} onClick={() => on && setHover({ t, s })} onMouseLeave={() => setHover(null)}>
                <rect x={x + 1} y={y + 1} width={cellW - 2} height={cellH - 2} rx={5}
                  fill={on ? PIE_COLORS[u] : '#f1f1f1'} opacity={on ? (reachable ? 0.85 : 0.35) : 1}
                  stroke={hover && hover.t === t && hover.s === s ? '#111' : 'white'} strokeWidth={hover && hover.t === t && hover.s === s ? 2 : 1} />
                {on && (
                  <text x={x + cellW / 2} y={y + cellH / 2 + 5} textAnchor="middle" className="cell-value" fill={u === 1 || u === 3 ? 'white' : '#222'}>
                    {sol.V[t][s].toFixed(0)}
                  </text>
                )}
              </g>
            );
          }),
        )}
      </svg>
      {step < 4 && <div className="bellman-detail">
        {q && shown(h.t) ? (
          <>
            <div>
              At period <b>{h.t}</b> with <b>{h.s}</b> cars, the four choices cost (now + expected later):
            </div>
            <div className="q-values">
              {q.map((v, u) => (
                <span key={u} className={`q-chip ${u === sol.u[h.t][h.s] ? 'best' : ''}`}>
                  <i style={{ background: PIE_COLORS[u] }} />
                  {ACTION_WORDS[u]}: <b>{v.toFixed(1)}</b>
                </span>
              ))}
            </div>
            <div className="muted">The booth picks the smallest; the cell shows that value and its colour.</div>
          </>
        ) : (
          <span className="muted">Scroll on: the table fills from the last period backwards. Then hover or tap a cell.</span>
        )}
      </div>}
      {step >= 4 && <GainChart />}
    </div>
  );
}

function GainChart() {
  const gain = attendantGain(sc.selfish);
  const { T, K } = BASE;
  const colors = ['#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'];
  return (
    <LineChart
      title="How much is an attendant worth to the booth?"
      xLabel="period t"
      yLabel="private gain"
      height={200}
      xTicks={Array.from({ length: T }, (_, t) => t)}
      series={Array.from({ length: K + 1 }, (_, s) => ({
        label: `${s} car${s === 1 ? '' : 's'}`,
        color: colors[s],
        marker: true,
        points: Array.from({ length: T }, (_, t) => [t, gain[t][s]] as [number, number]).filter(([t]) => t >= s),
      }))}
    />
  );
}

export function SectionBellman() {
  const sol = sc.selfish;
  const gains = attendantGain(sol);
  const reach = (t: number, s: number) => s <= t;
  const gainMax = Math.max(...gains.flatMap((row, t) => row.filter((_, s) => reach(t, s))));
  const gainEmpty = Math.max(...gains.map((row) => row[0]));
  const lastAll = sol.u[BASE.T - 1].slice(0, BASE.K).every((u) => ADMIT[u] === 1 && HIGH[u] === 0);
  return (
    <Scrolly
      id="bellman"
      number="6"
      title="Planning backwards"
      kicker="How can one booth find its best plan for the whole rush hour? Start at the end."
      steps={[
        {
          key: 'a',
          body: (
            <p>
              A <b>policy</b> is a plan: for each period <TeX>t</TeX> and each state <TeX>s</TeX>, which action to take. That is a table with 10 × 6 = 60 cells and 4 choices per cell, so there are <TeX>{'4^{60} \\approx 10^{36}'}</TeX> possible policies. Trying them all would take longer than the age of the universe.
            </p>
          ),
        },
        {
          key: 'b',
          body: (
            <p>
              Trick: <b>start at the end</b>. In the last period there is no future, so the booth just picks the cheapest action now: open the gate (avoid the missed toll) and no attendant (save 0.18). {lastAll ? 'That is the colour of the whole last column, except the full lane, where the gate makes no difference.' : ''}
            </p>
          ),
        },
        {
          key: 'c',
          body: (
            <>
              <p>
                One step earlier, the booth knows what it will do next, and what that will cost. So each action is worth "cost now + expected cost later". This is <b>Bellman's equation</b>:
              </p>
              <TeX block>{"V_t(s) = \\min_{u}\\Big\\{\\, c(s,u) + \\sum_{s'} P(s' \\mid s,u)\\, V_{t+1}(s') \\Big\\}"}</TeX>
              <p>
                Here <TeX>{'V_t(s)'}</TeX> is the smallest expected cost from period <TeX>t</TeX> to the end with <TeX>s</TeX> cars, and after the last period
                there is nothing left to pay: <TeX>{'V_T(s) = 0'}</TeX>. Watch the table fill from right to left: 240 small computations instead of <TeX>{'10^{36}'}</TeX>.
              </p>
            </>
          ),
        },
        {
          key: 'd',
          body: (
            <p>
              The result is the booth's best plan <b>for itself</b>. We call it the <b>selfish policy</b>: open the gate whenever the lane is not full, and call an attendant early in the rush hour, later only when the lane is long. Hover a cell to see the four options.
            </p>
          ),
        },
        {
          key: 'e',
          body: (
            <p>
              Why call attendants so often? An attendant is worth much more in a long lane (up to about {gainMax.toFixed(0)} cost units) than in an empty one (at most {gainEmpty.toFixed(1)}), because it speeds up <i>every</i> car. For one booth on its own, this plan is perfectly sensible. The trouble starts when <b>every</b> booth follows it.
            </p>
          ),
        },
      ]}
      render={(step) => <BellmanTable step={step} />}
    />
  );
}
