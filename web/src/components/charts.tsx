/** Small SVG charts: line chart and grouped bar chart, with readable defaults. */
import type { ReactNode } from 'react';

export interface Series {
  label: string;
  color: string;
  points: [number, number][];
  dash?: string;
  marker?: boolean;
  width?: number;
  band?: [number, number, number][]; // x, low, high
  line?: boolean; // false: markers only
}

function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (hi <= lo) return [lo];
  const span = hi - lo;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

export function LineChart({ series, title, xLabel, yLabel, xDomain, yDomain, height = 230, width = 520, xTicks, hLines = [], log = false, note }: {
  series: Series[];
  title?: ReactNode;
  xLabel?: string;
  yLabel?: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  height?: number;
  width?: number;
  xTicks?: number[];
  hLines?: { y: number; label: string; color: string; dash?: string; below?: boolean }[];
  log?: boolean;
  note?: ReactNode;
}) {
  const pad = { l: 52, r: 14, t: 12, b: 38 };
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => [...s.points.map((p) => p[1]), ...(s.band ?? []).flatMap((b) => [b[1], b[2]])]).concat(hLines.map((h) => h.y));
  const [x0, x1] = xDomain ?? [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = yDomain ?? [Math.min(...ys), Math.max(...ys)];
  if (!yDomain) {
    const m = (y1 - y0) * 0.08 || 1;
    y0 -= m;
    y1 += m;
  }
  const fx = (x: number) => pad.l + ((log ? Math.log10(x) - Math.log10(x0) : x - x0) / (log ? Math.log10(x1) - Math.log10(x0) : x1 - x0 || 1)) * (width - pad.l - pad.r);
  const fy = (y: number) => pad.t + (1 - (y - y0) / (y1 - y0 || 1)) * (height - pad.t - pad.b);
  const yt = niceTicks(y0, y1);
  const xt = xTicks ?? niceTicks(x0, x1, 6);
  return (
    <div className="chart">
      {title && <div className="chart-title-html">{title}</div>}
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        {yt.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={width - pad.r} y1={fy(v)} y2={fy(v)} stroke="#e7e7e7" />
            <text x={pad.l - 6} y={fy(v) + 4} textAnchor="end" className="axis-tick">{Math.abs(v) >= 1000 ? v.toExponential(0) : v}</text>
          </g>
        ))}
        {xt.map((v) => (
          <text key={v} x={fx(v)} y={height - pad.b + 16} textAnchor="middle" className="axis-tick">{v >= 1000 ? `${v / 1000}k` : v}</text>
        ))}
        <line x1={pad.l} x2={width - pad.r} y1={height - pad.b} y2={height - pad.b} stroke="#999" />
        {xLabel && <text x={(pad.l + width - pad.r) / 2} y={height - 4} textAnchor="middle" className="axis-label">{xLabel}</text>}
        {yLabel && <text transform={`translate(12, ${(pad.t + height - pad.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">{yLabel}</text>}
        {hLines.map((h) => (
          <g key={h.label}>
            <line x1={pad.l} x2={width - pad.r} y1={fy(h.y)} y2={fy(h.y)} stroke={h.color} strokeDasharray={h.dash ?? '5 4'} strokeWidth={1.4} />
            <text x={width - pad.r - 4} y={fy(h.y) + (h.below ? 13 : -5)} textAnchor="end" className="axis-tick" fill={h.color}>{h.label}</text>
          </g>
        ))}
        {series.map((s) => (
          <g key={s.label}>
            {s.band && (
              <path
                d={`M ${s.band.map((b) => `${fx(b[0])} ${fy(b[2])}`).join(' L ')} L ${[...s.band].reverse().map((b) => `${fx(b[0])} ${fy(b[1])}`).join(' L ')} Z`}
                fill={s.color}
                opacity={0.15}
              />
            )}
            {s.line !== false && (
              <polyline points={s.points.map((p) => `${fx(p[0])},${fy(p[1])}`).join(' ')} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash} />
            )}
            {s.marker && s.points.map((p, i) => <circle key={i} cx={fx(p[0])} cy={fy(p[1])} r={3} fill={s.color} />)}
          </g>
        ))}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.label} className="legend-item">
            <i style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      {note && <div className="chart-note">{note}</div>}
    </div>
  );
}

export function BarChart({ groups, series, title, yLabel, height = 220, yMax, format = (v: number) => v.toFixed(2), note }: {
  groups: string[];
  series: { label: string; color: string; values: number[]; outline?: boolean }[];
  title?: ReactNode;
  yLabel?: string;
  height?: number;
  yMax?: number;
  format?: (v: number) => string;
  note?: ReactNode;
}) {
  const width = 520;
  const pad = { l: 46, r: 10, t: 14, b: 34 };
  const max = yMax ?? Math.max(1e-9, ...series.flatMap((s) => s.values)) * 1.12;
  const min = Math.min(0, ...series.flatMap((s) => s.values)) * 1.25;
  const gw = (width - pad.l - pad.r) / groups.length;
  const bw = (gw * 0.8) / series.length;
  const fy = (v: number) => pad.t + (1 - (v - min) / (max - min)) * (height - pad.t - pad.b);
  const yt = niceTicks(min, max, 4);
  return (
    <div className="chart">
      {title && <div className="chart-title-html">{title}</div>}
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        {yt.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={width - pad.r} y1={fy(v)} y2={fy(v)} stroke="#e7e7e7" />
            <text x={pad.l - 6} y={fy(v) + 4} textAnchor="end" className="axis-tick">{v}</text>
          </g>
        ))}
        {groups.map((g, gi) => (
          <g key={g}>
            {series.map((s, si) => {
              const v = s.values[gi] ?? 0;
              const x = pad.l + gi * gw + gw * 0.1 + si * bw;
              return (
                <g key={s.label}>
                  <rect x={x} y={Math.min(fy(v), fy(0))} width={bw * 0.92} height={Math.abs(fy(0) - fy(v))} fill={s.outline ? 'none' : s.color} stroke={s.color} strokeWidth={s.outline ? 2 : 0} rx={2} />
                  {v !== 0 && series.length <= 3 && groups.length * series.length <= 8 && (
                    <text x={x + bw * 0.46} y={v > 0 ? fy(v) - 4 : fy(v) + 12} textAnchor="middle" className="axis-tick">{format(v)}</text>
                  )}
                </g>
              );
            })}
            <text x={pad.l + gi * gw + gw / 2} y={height - pad.b + 16} textAnchor="middle" className="axis-tick">{g}</text>
          </g>
        ))}
        <line x1={pad.l} x2={width - pad.r} y1={fy(0)} y2={fy(0)} stroke="#999" />
        {yLabel && <text transform={`translate(12, ${(pad.t + height - pad.b) / 2}) rotate(-90)`} textAnchor="middle" className="axis-label">{yLabel}</text>}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.label} className="legend-item">
            <i style={{ background: s.outline ? 'transparent' : s.color, border: `2px solid ${s.color}` }} />
            {s.label}
          </span>
        ))}
      </div>
      {note && <div className="chart-note">{note}</div>}
    </div>
  );
}
