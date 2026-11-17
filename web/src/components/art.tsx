/**
 * Drawing primitives of the toll plaza (SVG, top view): cars, booths, gates, attendants, request bubbles and lane
 * asphalt. Everything is drawn around (0, 0), or from a top-left corner where stated, so callers position it with a
 * transform. Moving things are memoised sprites that glide with a CSS transition.
 */
import { memo, type CSSProperties, type ReactNode } from 'react';

export const CAR_COLORS = ['#e4572e', '#29335c', '#f3a712', '#669bbc', '#a8c686', '#db3a34', '#7b4b94', '#2a9d8f', '#ef8354', '#5c80bc'];
export const ATTENDANT_COLOR = '#0072B2';
export const COORD_COLOR = '#0072B2';
export const SELFISH_COLOR = '#009E73';
export const CONGESTION_COLOR = '#CC79A7';
export const PRICE_COLOR = '#E69F00';
export const LOST_COLOR = '#c0392b';
export const PIE_COLORS = ['#f9a73e', '#bf212f', '#27b376', '#006f3c']; // (a,b) = 00, 01, 10, 11, as in the research paper
export const ACTION_WORDS = ['gate closed, normal speed', 'gate closed, attendant', 'gate open, normal speed', 'gate open, attendant'];

/** A car seen from above, pointing right, centred at the origin. */
export const Car = memo(function Car({ w, h, color, tone }: { w: number; h: number; color: string; tone?: 'lost' | null }) {
  const body = tone === 'lost' ? LOST_COLOR : color;
  return (
    <g>
      <rect x={-w / 2} y={-h / 2} width={w} height={h} rx={h * 0.32} fill={body} stroke="rgba(0,0,0,0.35)" strokeWidth={Math.max(0.6, h * 0.04)} />
      <rect x={w * 0.02} y={-h * 0.36} width={w * 0.2} height={h * 0.72} rx={h * 0.12} fill="rgba(210,235,255,0.85)" />
      <rect x={-w * 0.36} y={-h * 0.33} width={w * 0.14} height={h * 0.66} rx={h * 0.1} fill="rgba(210,235,255,0.6)" />
      {tone === 'lost' && (
        <g stroke="white" strokeWidth={h * 0.1} strokeLinecap="round">
          <line x1={-h * 0.22} y1={-h * 0.22} x2={h * 0.22} y2={h * 0.22} />
          <line x1={-h * 0.22} y1={h * 0.22} x2={h * 0.22} y2={-h * 0.22} />
        </g>
      )}
    </g>
  );
});

/**
 * A toll booth seen from above: a small square cabin with a red roof edge and a window facing the lane.
 * `x, y` is the top-left corner; `size` is its side. A `missed` booth (no new car this period) shows a red window.
 */
export const Booth = memo(function Booth({ x, y, size, missed = false }: { x: number; y: number; size: number; missed?: boolean }) {
  const r = Math.min(6, size * 0.16);
  return (
    <g transform={`translate(${x},${y})`}>
      <rect x={0} y={0} width={size} height={size} rx={r} fill="#f2c14e" stroke="#8a6d1d" strokeWidth={Math.max(0.8, size * 0.03)} />
      <rect x={0} y={0} width={size} height={size * 0.18} rx={r} fill="#d1495b" />
      <rect
        x={size * 0.14}
        y={size * 0.34}
        width={size * 0.42}
        height={size * 0.44}
        rx={Math.max(1, size * 0.06)}
        fill={missed ? LOST_COLOR : '#fff7dc'}
        stroke="#8a6d1d"
        strokeWidth={Math.max(0.5, size * 0.02)}
        style={{ transition: 'fill 250ms ease' }}
      />
    </g>
  );
});

/**
 * A red-and-white barrier at the entrance of a lane. The pivot is at (x, y), the bottom edge of the lane, and the arm
 * spans the lane (`len`). It lifts by rotating about the pivot.
 */
export const Gate = memo(function Gate({ x, y, len, open, dur = 280 }: { x: number; y: number; len: number; open: boolean; dur?: number }) {
  const width = Math.min(6, Math.max(2, len * 0.11));
  const style: CSSProperties = {
    transform: `translate(${x}px, ${y}px) rotate(${open ? -78 : 0}deg)`,
    transition: dur > 0 ? `transform ${dur}ms ease-in-out` : undefined,
  };
  return (
    <g style={style}>
      <line x1={0} y1={0} x2={0} y2={-len} stroke="white" strokeWidth={width} strokeLinecap="round" />
      <line x1={0} y1={0} x2={0} y2={-len} stroke="#d1495b" strokeWidth={width} strokeDasharray={`${Math.max(2, len * 0.14)} ${Math.max(2, len * 0.12)}`} />
      <circle r={Math.max(1.6, Math.min(4, len * 0.08))} fill="#3a3a3a" />
    </g>
  );
});

/** A mobile attendant (a little person with a card machine), centred at the origin. */
export const Attendant = memo(function Attendant({ size, shared = false }: { size: number; shared?: boolean }) {
  return (
    <g opacity={shared ? 0.6 : 1}>
      <circle cx={0} cy={-size * 0.3} r={size * 0.2} fill="#f1c27d" stroke="#5a3e1b" strokeWidth={size * 0.03} />
      <path d={`M ${-size * 0.3} ${size * 0.42} Q 0 ${-size * 0.3} ${size * 0.3} ${size * 0.42} Z`} fill={ATTENDANT_COLOR} />
      <rect x={size * 0.16} y={0} width={size * 0.2} height={size * 0.26} rx={size * 0.04} fill="#222" />
      {shared && size >= 16 && (
        <text x={0} y={size * 0.85} fontSize={size * 0.36} textAnchor="middle" fill={ATTENDANT_COLOR} fontWeight={700}>
          shared
        </text>
      )}
    </g>
  );
});

/** A speech bubble with what a booth asks for: a car, an attendant, or both. Drawn from its left-centre point. */
export const RequestBubble = memo(function RequestBubble({ x, y, h, car, attendant }: { x: number; y: number; h: number; car: boolean; attendant: boolean }) {
  if (!car && !attendant) return null;
  const icon = h * 0.62;
  const n = (car ? 1 : 0) + (attendant ? 1 : 0);
  const w = h * 0.5 + n * icon * 1.6;
  return (
    <g transform={`translate(${x},${y})`}>
      <g className="bubble-pop">
      <path d={`M 0 0 L ${h * 0.28} ${-h * 0.22} L ${h * 0.28} ${h * 0.22} Z`} fill="white" stroke="#777" strokeWidth={0.7} />
      <rect x={h * 0.22} y={-h / 2} width={w} height={h} rx={h * 0.45} fill="white" stroke="#777" strokeWidth={0.7} />
      {car && (
        <g transform={`translate(${h * 0.47 + icon * 0.8}, 0)`}>
          <Car w={icon * 1.4} h={icon * 0.7} color="#669bbc" />
        </g>
      )}
      {attendant && (
        <g transform={`translate(${h * 0.47 + (car ? icon * 1.6 : 0) + icon * 0.8}, ${icon * 0.05})`}>
          <Attendant size={icon * 1.05} />
        </g>
      )}
      </g>
    </g>
  );
});

/** Asphalt of one lane with slot marks; `x0..x1` horizontally, centred on `cy`. */
export const LaneStrip = memo(function LaneStrip({ x0, x1, cy, h, slots, highlight }: { x0: number; x1: number; cy: number; h: number; slots: number; highlight?: boolean }) {
  const slotW = (x1 - x0) / slots;
  const dash = Math.max(4, Math.min(10, h * 0.4));
  return (
    <g>
      <rect x={x0} y={cy - h / 2} width={x1 - x0} height={h} fill={highlight ? '#565d68' : '#4a4f57'} />
      {Array.from({ length: slots - 1 }, (_, k) => (
        <line key={k} x1={x0 + (k + 1) * slotW} x2={x0 + (k + 1) * slotW} y1={cy - h / 2 + 2} y2={cy + h / 2 - 2} stroke="rgba(255,255,255,0.1)" strokeWidth={1} />
      ))}
      <line x1={x0} x2={x1} y1={cy - h / 2} y2={cy - h / 2} stroke="white" strokeWidth={Math.min(1.2, h * 0.06)} strokeDasharray={`${dash} ${dash * 0.8}`} opacity={0.65} />
      <line x1={x0} x2={x1} y1={cy + h / 2} y2={cy + h / 2} stroke="white" strokeWidth={Math.min(1.2, h * 0.06)} strokeDasharray={`${dash} ${dash * 0.8}`} opacity={0.65} />
    </g>
  );
});

/** A sprite that glides to its position (and opacity) with a CSS transition of `dur` milliseconds. */
export function Sprite({ x, y, dur, opacity = 1, children }: { x: number; y: number; dur: number; opacity?: number; children: ReactNode }) {
  return (
    <g
      style={{
        transform: `translate(${x}px, ${y}px)`,
        transition: dur > 0 ? `transform ${dur}ms ease-in-out, opacity ${dur}ms ease-in-out` : undefined,
        opacity,
      }}
    >
      {children}
    </g>
  );
}

/** A memoised car sprite: only re-renders when its own position, look or timing changes. */
export const CarSprite = memo(function CarSprite({ x, y, dur, opacity, w, h, color, lost }: {
  x: number;
  y: number;
  dur: number;
  opacity: number;
  w: number;
  h: number;
  color: string;
  lost?: boolean;
}) {
  return (
    <Sprite x={x} y={y} dur={dur} opacity={opacity}>
      <Car w={w} h={h} color={color} tone={lost ? 'lost' : null} />
    </Sprite>
  );
});

/** A memoised attendant sprite. */
export const AttendantSprite = memo(function AttendantSprite({ x, y, dur, opacity, size, shared }: {
  x: number;
  y: number;
  dur: number;
  opacity: number;
  size: number;
  shared?: boolean;
}) {
  return (
    <Sprite x={x} y={y} dur={dur} opacity={opacity}>
      <Attendant size={size} shared={shared} />
    </Sprite>
  );
});

/**
 * Run `cb` once the browser has painted the current frame, so that a CSS transition starts from what is on screen.
 * A timer is the fallback when frames are throttled (background tabs), so the sequence never stalls.
 */
export function afterPaint(cb: () => void): void {
  let done = false;
  const run = () => {
    if (!done) {
      done = true;
      cb();
    }
  };
  requestAnimationFrame(() => requestAnimationFrame(run));
  window.setTimeout(run, 60);
}
