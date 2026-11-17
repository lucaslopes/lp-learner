import type { ReactNode } from 'react';

export function Toggle<T extends string | number>({ label, value, options, onChange }: {
  label?: ReactNode;
  value: T;
  options: { value: T; label: ReactNode; title?: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="toggle">
      {label && <span className="control-label">{label}</span>}
      <div className="toggle-group" role="radiogroup">
        {options.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={o.value === value}
            title={o.title}
            className={o.value === value ? 'on' : ''}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Slider({ label, value, min, max, step = 1, onChange, format = (v) => String(v), marks }: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  marks?: number[];
}) {
  // With `marks`, the slider moves over the index of the allowed values.
  if (marks) {
    const idx = Math.max(0, marks.findIndex((m) => Math.abs(m - value) < 1e-9));
    return (
      <label className="slider">
        <span className="control-label">
          {label} <b>{format(value)}</b>
        </span>
        <input type="range" min={0} max={marks.length - 1} step={1} value={idx} onChange={(e) => onChange(marks[Number(e.target.value)])} />
      </label>
    );
  }
  return (
    <label className="slider">
      <span className="control-label">
        {label} <b>{format(value)}</b>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

export function Button({ children, onClick, kind = 'default', disabled, title }: {
  children: ReactNode;
  onClick: () => void;
  kind?: 'default' | 'primary' | 'ghost';
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button type="button" className={`btn btn-${kind}`} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  );
}

/** A small "how does this work?" link that scrolls back to the section that explains a panel. */
export function Explain({ to, children = 'How does this work?' }: { to: string; children?: ReactNode }) {
  return (
    <a
      className="explain"
      href={`#${to}`}
      onClick={(e) => {
        e.preventDefault();
        document.getElementById(to)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }}
    >
      ↑ {children}
    </a>
  );
}

export function Stat({ label, value, sub, tone }: { label: ReactNode; value: ReactNode; sub?: ReactNode; tone?: 'coord' | 'selfish' | 'warn' }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ''}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}
