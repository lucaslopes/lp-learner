/**
 * The pie plot of the research paper: at each period t (horizontal) and lane length s (vertical), the area of the pie
 * is the fraction of booths in that state (normalised per period, shares below 1% not drawn) and its slices split
 * those booths by action (a, b), in the paper's colours.
 */
import { useState } from 'react';
import { ACTION_WORDS, PIE_COLORS } from './art';

function wedge(cx: number, cy: number, r: number, a0: number, a1: number): string {
  if (a1 - a0 >= 2 * Math.PI - 1e-9) return `M ${cx - r} ${cy} a ${r} ${r} 0 1 0 ${2 * r} 0 a ${r} ${r} 0 1 0 ${-2 * r} 0`;
  const x0 = cx + r * Math.cos(a0);
  const y0 = cy - r * Math.sin(a0);
  const x1 = cx + r * Math.cos(a1);
  const y1 = cy - r * Math.sin(a1);
  const large = a1 - a0 > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${x0} ${y0} A ${r} ${r} 0 ${large} 0 ${x1} ${y1} Z`;
}

export interface PieHover {
  t: number;
  s: number;
  share: number;
  split: number[];
}

export function PiePlot({ y, title, highlightT, legend = true, compact = false }: {
  y: number[][][];
  title?: string;
  highlightT?: number | null;
  legend?: boolean;
  compact?: boolean;
}) {
  const T = y.length;
  const S = y[0].length;
  const cell = 52;
  const padL = 44;
  const padB = 40;
  const padT = title ? 30 : 12;
  const width = padL + T * cell + 10;
  const height = padT + S * cell + padB;
  const [hover, setHover] = useState<PieHover | null>(null);
  const cx = (t: number) => padL + (t + 0.5) * cell;
  const cy = (s: number) => padT + (S - s - 0.5) * cell;
  const maxR = cell * 0.62;

  const pies = [];
  for (let t = 0; t < T; t++) {
    const masses = y[t].map((row) => row.reduce((acc, v) => acc + Math.max(v, 0), 0));
    const total = masses.reduce((a, b) => a + b, 0);
    if (total <= 0) continue;
    for (let s = 0; s < S; s++) {
      const share = masses[s] / total;
      if (share < 0.01) continue;
      const r = maxR * Math.sqrt(share);
      const split = y[t][s].map((v) => Math.max(v, 0) / masses[s]);
      let a = Math.PI / 2;
      const paths = split.map((f, u) => {
        if (f <= 1e-4) return null;
        const a1 = a + f * 2 * Math.PI;
        const d = wedge(cx(t), cy(s), r, a, a1);
        a = a1;
        return <path key={u} d={d} fill={PIE_COLORS[u]} stroke="white" strokeWidth={0.6} />;
      });
      pies.push(
        <g
          key={`${t}-${s}`}
          onMouseEnter={() => setHover({ t, s, share, split })}
          onClick={() => setHover({ t, s, share, split })}
          onMouseLeave={() => setHover(null)}
          opacity={highlightT == null || highlightT === t ? 1 : 0.35}
          style={{ cursor: 'default' }}
        >
          <circle cx={cx(t)} cy={cy(s)} r={Math.max(r, 6)} fill="transparent" />
          {paths}
        </g>,
      );
    }
  }

  return (
    <div className={`pie-plot ${compact ? 'compact' : ''}`}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title ?? 'Pie plot of a policy'}>
        {title && <text x={padL} y={18} className="chart-title">{title}</text>}
        {Array.from({ length: S }, (_, s) => (
          <g key={s}>
            <line x1={padL} x2={padL + T * cell} y1={cy(s)} y2={cy(s)} stroke="#eee" />
            <text x={padL - 10} y={cy(s) + 4} textAnchor="end" className="axis-tick">{s}</text>
          </g>
        ))}
        {Array.from({ length: T }, (_, t) => (
          <text key={t} x={cx(t)} y={padT + S * cell + 16} textAnchor="middle" className="axis-tick">{t}</text>
        ))}
        <text x={padL + (T * cell) / 2} y={height - 4} textAnchor="middle" className="axis-label">period t</text>
        <text transform={`translate(12, ${padT + (S * cell) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">cars in the lane s</text>
        {pies}
      </svg>
      <div className="pie-tooltip" aria-live="polite">
        {hover ? (
          <>
            Period {hover.t}, {hover.s} car{hover.s === 1 ? '' : 's'}: <b>{(100 * hover.share).toFixed(0)}%</b> of the booths.{' '}
            {hover.split.map((f, u) => (f > 0.005 ? `${(100 * f).toFixed(0)}% ${ACTION_WORDS[u]}` : null)).filter(Boolean).join(' · ')}
          </>
        ) : (
          <span className="muted">Hover or tap a pie: its size is the share of booths in that state, its slices what they do.</span>
        )}
      </div>
      {legend && <PieLegend />}
    </div>
  );
}

export function PieLegend() {
  return (
    <div className="legend">
      {ACTION_WORDS.map((w, u) => (
        <span key={u} className="legend-item">
          <i style={{ background: PIE_COLORS[u] }} />
          {w}
        </span>
      ))}
    </div>
  );
}
