/**
 * One booth and its lane, drawn large for the single-booth sections, and `useLane`, a small state machine that animates
 * what happens to its cars: a car drives through the lifting gate to the end of the queue; a paying car drives out
 * through the booth while the queue moves up; a car that cannot get in (gate closed or lane full) turns red and drives
 * away. Cars are listed front first.
 */
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { Attendant, Booth, CAR_COLORS, CarSprite, Gate, LaneStrip, Sprite, afterPaint } from './art';

export type CarPhase = 'spawn' | 'queued' | 'leaving' | 'lost';

export interface LaneCar {
  id: number;
  color: string;
  phase: CarPhase;
}

export const LANE = { x0: 160, x1: 530, cy: 104, h: 74, K: 5 };
const ENTER_MS = 750;
const LEAVE_MS = 650;

let serial = 1;
function makeCar(phase: CarPhase = 'queued'): LaneCar {
  const id = serial++;
  return { id, color: CAR_COLORS[id % CAR_COLORS.length], phase };
}

export function slotX(slot: number, K = LANE.K) {
  const slotW = (LANE.x1 - LANE.x0) / K;
  return LANE.x1 - (slot + 0.5) * slotW;
}

/** Cars of a lane with animated arrivals and departures. */
export function useLane(initial: number, K = LANE.K) {
  const [cars, setCars] = useState<LaneCar[]>(() => Array.from({ length: initial }, () => makeCar()));
  const [gateLifted, setGateLifted] = useState(false);
  const timers = useRef<number[]>([]);
  const carsRef = useRef(cars);
  carsRef.current = cars;
  useEffect(() => () => timers.current.forEach((t) => window.clearTimeout(t)), []);
  const later = useCallback((ms: number, fn: () => void) => {
    timers.current.push(window.setTimeout(fn, ms));
  }, []);
  const setPhase = useCallback((ids: Set<number>, phase: CarPhase) => setCars((cs) => cs.map((c) => (ids.has(c.id) ? { ...c, phase } : c))), []);
  const remove = useCallback((ids: Set<number>) => setCars((cs) => cs.filter((c) => !ids.has(c.id))), []);
  const queued = cars.filter((c) => c.phase === 'queued').length;

  /** A car arrives. It enters through the lifting gate if `canEnter` and there is room; otherwise it is turned away. */
  const arrive = useCallback(
    (canEnter = true): Promise<'entered' | 'lost'> =>
      new Promise((resolve) => {
        const room = carsRef.current.filter((c) => c.phase === 'queued' || c.phase === 'spawn').length < K;
        const car = makeCar('spawn');
        setCars((cs) => [...cs, car]);
        afterPaint(() => {
          if (canEnter && room) {
            setGateLifted(true);
            setPhase(new Set([car.id]), 'queued');
            later(ENTER_MS, () => {
              setGateLifted(false);
              resolve('entered');
            });
          } else {
            setPhase(new Set([car.id]), 'lost');
            later(ENTER_MS, () => {
              remove(new Set([car.id]));
              resolve('lost');
            });
          }
        });
      }),
    [K, later, remove, setPhase],
  );

  /** These cars pay and drive out through the booth; the queue moves up. */
  const pay = useCallback(
    (ids: number[]): Promise<void> =>
      new Promise((resolve) => {
        if (!ids.length) return resolve();
        const set = new Set(ids);
        setPhase(set, 'leaving');
        later(LEAVE_MS, () => {
          remove(set);
          resolve();
        });
      }),
    [later, remove, setPhase],
  );

  /** The first car in the queue pays. */
  const payFirst = useCallback(() => {
    const first = carsRef.current.find((c) => c.phase === 'queued');
    return pay(first ? [first.id] : []);
  }, [pay]);

  /** Bring the lane to `n` cars, animating the cars that join or leave. */
  const setCount = useCallback(
    (n: number) => {
      const q = carsRef.current.filter((c) => c.phase === 'queued');
      if (n > q.length) {
        const added = Array.from({ length: n - q.length }, () => makeCar('spawn'));
        setCars((cs) => [...cs.filter((c) => c.phase !== 'leaving'), ...added]);
        afterPaint(() => setPhase(new Set(added.map((c) => c.id)), 'queued'));
      } else if (n < q.length) {
        void pay(q.slice(0, q.length - n).map((c) => c.id));
      }
    },
    [pay, setPhase],
  );

  /** Replace the lane by `n` fresh cars without animation. */
  const reset = useCallback((n: number) => setCars(Array.from({ length: n }, () => makeCar())), []);

  /** The cars waiting in the queue right now (front first), for code that runs across several frames. */
  const waiting = useCallback(() => carsRef.current.filter((c) => c.phase === 'queued'), []);

  return { cars, queued, gateLifted, arrive, pay, payFirst, setCount, reset, waiting };
}

export function LaneView({ cars, gateOpen, attendant, showSlots = false, full = false, children, caption }: {
  cars: LaneCar[];
  gateOpen: boolean;
  attendant: boolean;
  showSlots?: boolean;
  full?: boolean;
  children?: ReactNode;
  caption?: ReactNode;
}) {
  const K = LANE.K;
  const carH = 30;
  const carW = 58;
  let queued = 0;
  return (
    <div className="lane-view">
      <svg viewBox="0 0 660 210" role="img" aria-label="A toll booth and its lane">
        <rect x={0} y={0} width={660} height={210} rx={16} fill="#e8eee4" />
        <rect x={0} y={LANE.cy - LANE.h / 2 - 4} width={LANE.x0 - 8} height={LANE.h + 8} fill="#cfd6ca" rx={8} />
        <text x={14} y={LANE.cy - LANE.h / 2 - 10} className="svg-label">road to the plaza</text>
        <LaneStrip x0={LANE.x0} x1={LANE.x1} cy={LANE.cy} h={LANE.h} slots={K} highlight={full} />
        {showSlots &&
          Array.from({ length: K }, (_, k) => (
            <text key={k} x={slotX(k)} y={LANE.cy + LANE.h / 2 + 18} textAnchor="middle" className="svg-label">
              place {k + 1}
            </text>
          ))}
        <Booth x={LANE.x1 + 6} y={LANE.cy - 32} size={64} />
        <text x={LANE.x1 + 38} y={LANE.cy + 52} textAnchor="middle" className="svg-label">booth</text>
        <Gate x={LANE.x0 - 3} y={LANE.cy + LANE.h / 2} len={LANE.h} open={gateOpen} />
        <Sprite x={attendant ? LANE.x1 + 98 : 700} y={LANE.cy} dur={450} opacity={attendant ? 1 : 0}>
          <Attendant size={42} />
        </Sprite>
        {cars.map((c) => {
          let x: number;
          let opacity = 1;
          let dur = c.phase === 'leaving' ? LEAVE_MS : ENTER_MS;
          if (c.phase === 'spawn') {
            x = 40;
            opacity = 0;
            dur = 0;
          } else if (c.phase === 'leaving') {
            x = 700;
            opacity = 0;
          } else if (c.phase === 'lost') {
            x = LANE.x0 - 40;
            opacity = 0;
          } else x = slotX(queued++);
          return <CarSprite key={c.id} x={x} y={LANE.cy} dur={dur} opacity={opacity} w={carW} h={carH} color={c.color} lost={c.phase === 'lost'} />;
        })}
        {children}
      </svg>
      {caption && <div className="lane-caption">{caption}</div>}
    </div>
  );
}
