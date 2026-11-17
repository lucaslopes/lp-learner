/**
 * The animated toll plaza. A Plaza (engine/sim) is stepped one period at a time and every step is played out:
 *
 *   1. the platoon drives in and waits at the entrance;
 *   2. booths say what they want (speech bubbles);
 *   3. the lotteries decide: attendants walk to the winners and the winners' gates lift;
 *   4. cars drive through the open gates to the end of their lanes; the others turn red and drive away
 *      (a booth without a new car shows a red window: a missed toll); then the gates close;
 *   5. cars pay and leave through the booth, each lane on its own clock; the queues move up;
 *   6. after T periods the lanes empty and a new rush hour starts.
 *
 * At "turbo" speed (>= 12) the phases are skipped and each period is drawn once, so that many rush hours can be
 * simulated quickly. Car and attendant sprites are memoised, so a phase only re-renders the sprites that move.
 * The plaza pauses while it is off screen or the tab is hidden.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { capacities, type Model } from '../engine/model';
import { Plaza, type Controller, type PeriodOutcome, type Rule } from '../engine/sim';
import { AttendantSprite, Booth, CAR_COLORS, Car, CarSprite, Gate, LaneStrip, RequestBubble, afterPaint } from './art';

interface CarState {
  id: number;
  x: number;
  y: number;
  color: string;
  opacity: number;
  dur: number;
  lost?: boolean;
  w: number;
  h: number;
}

interface AttState {
  id: number;
  x: number;
  y: number;
  dur: number;
  opacity: number;
  shared?: boolean;
}

export interface PlazaStageProps {
  model: Model;
  N: number;
  rule: Rule;
  controller: Controller;
  playing: boolean;
  speed: number;
  seed?: number;
  resetKey?: unknown;
  showRequests?: boolean;
  onPeriod?: (out: PeriodOutcome) => void;
  onRushHour?: (costPerBooth: number, rushHour: number) => void;
  title?: string;
  height?: number | string;
  fit?: 'meet' | 'slice';
}

const W = 1000;
const H = 560;
const TOP = 62;
const BOTTOM = 26;
const GATE_X = 300;
const LANE_END = 806;
const TURBO = 12;

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const painted = () => new Promise<void>((resolve) => afterPaint(resolve));
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function geometry(N: number, K: number, capA: number, capB: number) {
  const avail = H - TOP - BOTTOM;
  const laneH = Math.min(70, avail / N);
  const y0 = TOP + (avail - laneH * N) / 2;
  const laneCy = (i: number) => y0 + (i + 0.5) * laneH;
  const asphalt = laneH * 0.86;
  const slotW = (LANE_END - GATE_X) / K;
  const carH = clamp(laneH * 0.56, 5, 34);
  const carW = Math.min(slotW * 0.78, carH * 2.15);
  const slotPos = (i: number, slot: number) => ({ x: LANE_END - (slot + 0.5) * slotW, y: laneCy(i) });
  const booth = clamp(laneH * 0.86, 9, 54);
  const boothX = LANE_END + 6;
  const boothRight = boothX + booth;
  const att = clamp(laneH * 0.82, 10, 26);
  const attPos = (i: number) => ({ x: boothRight + att * 0.6, y: laneCy(i) });
  const bubbleX = boothRight + att * 1.15 + 2;
  const bubbleH = clamp(laneH * 0.62, 8, 22);
  const exitX = W + 60;
  // the platoon waits as a grid in the approach area
  const zx0 = 18;
  const zx1 = GATE_X - 46;
  const zy0 = TOP + 10;
  const zy1 = H - BOTTOM - 4;
  // the largest car size that fits, then the most compact grid (about 8 rows at most) with that size
  let hBest = 0;
  for (let cols = 1; cols <= Math.max(1, capA); cols++) {
    const rows = Math.ceil(capA / cols);
    hBest = Math.max(hBest, Math.min((zx1 - zx0) / cols / 2.4, (zy1 - zy0) / rows / 1.45, carH));
  }
  let best = { cols: 1, rows: Math.max(1, capA), h: hBest };
  for (let cols = 1; cols <= Math.max(1, capA); cols++) {
    const rows = Math.ceil(capA / cols);
    const fits = rows * hBest * 1.45 <= zy1 - zy0 + 1e-6 && cols * hBest * 2.4 <= zx1 - zx0 + 1e-6;
    if (fits) {
      best = { cols, rows, h: hBest };
      if (rows <= Math.max(8, Math.ceil(Math.sqrt(capA * 2)))) break;
    }
  }
  const platoonPos = (k: number) => {
    const col = k % best.cols;
    const row = Math.floor(k / best.cols);
    const gridH = best.rows * best.h * 1.45;
    const gridW = best.cols * best.h * 2.6;
    return { x: (zx0 + zx1) / 2 - gridW / 2 + (col + 0.5) * best.h * 2.6, y: (zy0 + zy1) / 2 - gridH / 2 + (row + 0.5) * best.h * 1.45 };
  };
  const roomStep = Math.min(34, 400 / Math.max(1, capB));
  const roomPos = (k: number) => ({ x: 580 + (k + 0.5) * roomStep, y: 39 });
  return {
    laneH, laneCy, asphalt, slotW, carH, carW, slotPos, booth, boothX, att, attPos, bubbleX, bubbleH, exitX, platoonPos,
    pW: best.h * 2.15, pH: best.h, roomPos,
  };
}

export function PlazaStage({
  model, N, rule, controller, playing, speed, seed = 2026, resetKey, showRequests = true, onPeriod, onRushHour, title, height = '100%', fit = 'meet',
}: PlazaStageProps) {
  const K = model.params.K;
  const [capA, capB] = capacities(N, model.budgets[0], model.budgets[1]);
  const geo = useMemo(() => geometry(N, K, capA, capB), [N, K, capA, capB]);

  const [cars, setCars] = useState<Record<number, CarState>>({});
  const [atts, setAtts] = useState<AttState[]>([]);
  const [gates, setGates] = useState<boolean[]>(() => new Array(N).fill(false));
  const [bubbles, setBubbles] = useState<{ a: number; b: number }[] | null>(null);
  const [missed, setMissed] = useState<boolean[] | null>(null);
  const [hud, setHud] = useState({ t: 0, rushHour: 0, banner: '' });

  const carsRef = useRef<Record<number, CarState>>({});
  const laneCarsRef = useRef<number[][]>([]);
  const nextId = useRef(1);
  const controllerRef = useRef(controller);
  const playingRef = useRef(playing);
  const speedRef = useRef(speed);
  const onPeriodRef = useRef(onPeriod);
  const onRushHourRef = useRef(onRushHour);
  const genRef = useRef(0);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const visibleRef = useRef(true);
  controllerRef.current = controller;
  playingRef.current = playing;
  speedRef.current = speed;
  onPeriodRef.current = onPeriod;
  onRushHourRef.current = onRushHour;

  // Animate only while the plaza is on screen and the tab is visible (several plazas live on the page).
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return undefined;
    const io = new IntersectionObserver((entries) => {
      visibleRef.current = entries.some((e) => e.isIntersecting);
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    const gen = ++genRef.current;
    const alive = () => genRef.current === gen;
    const plaza = new Plaza(model, N, rule, seed);
    const roomAll = (dur: number) => Array.from({ length: capB }, (_, k) => ({ id: k, ...geo.roomPos(k), dur, opacity: 1 }));
    carsRef.current = {};
    laneCarsRef.current = Array.from({ length: N }, () => []);
    setCars({});
    setGates(new Array(N).fill(false));
    setBubbles(null);
    setMissed(null);
    setHud({ t: 0, rushHour: 0, banner: '' });
    setAtts(roomAll(0));
    const turbo = () => speedRef.current >= TURBO;
    const d = (base: number) => (turbo() ? 0 : base / speedRef.current);
    const render = () => setCars({ ...carsRef.current });
    const move = (id: number, patch: Partial<CarState>) => {
      const c = carsRef.current[id];
      if (c) carsRef.current[id] = { ...c, ...patch };
    };
    const drop = (ids: number[]) => ids.forEach((id) => delete carsRef.current[id]);
    const newCar = (x: number, y: number, w: number, h: number, dur = 0): number => {
      const id = nextId.current++;
      carsRef.current[id] = { id, x, y, color: CAR_COLORS[id % CAR_COLORS.length], opacity: 1, dur, w, h };
      return id;
    };
    const attendantsFor = (out: PeriodOutcome, dur: number): AttState[] => {
      const callers = out.lanes.flatMap((l, i) => (l.b ? [i] : []));
      if (rule === 'congestion' && callers.length > capB) {
        // the attendants share their time among every caller
        return [
          ...Array.from({ length: capB }, (_, k) => ({ id: k, ...geo.roomPos(k), dur, opacity: 0 })),
          ...callers.map((i, k) => ({ id: 1000 + k, ...geo.attPos(i), dur: 0, opacity: 1, shared: true })),
        ];
      }
      return Array.from({ length: capB }, (_, k) => {
        const lane = callers[k];
        return { id: k, ...(lane === undefined ? geo.roomPos(k) : geo.attPos(lane)), dur, opacity: 1 };
      });
    };
    let costAcc = 0;

    const endOfPeriod = async (out: PeriodOutcome) => {
      costAcc += out.meanCost;
      onPeriodRef.current?.(out);
      if (plaza.t !== 0) return;
      // end of the rush hour: lanes empty, a new rush hour starts
      onRushHourRef.current?.(costAcc, out.rushHour);
      costAcc = 0;
      if (turbo()) {
        laneCarsRef.current.forEach((ids) => drop(ids));
        laneCarsRef.current = Array.from({ length: N }, () => []);
        render();
        return;
      }
      setHud((h) => ({ ...h, banner: `Rush hour ${out.rushHour + 1} over` }));
      laneCarsRef.current.forEach((ids) => ids.forEach((id) => move(id, { x: geo.exitX, opacity: 0, dur: d(650) })));
      render();
      setAtts(roomAll(d(500)));
      await sleep(d(1000));
      if (!alive()) return;
      laneCarsRef.current.forEach((ids) => drop(ids));
      laneCarsRef.current = Array.from({ length: N }, () => []);
      setHud((h) => ({ ...h, banner: '' }));
      render();
    };

    /** Turbo: draw the end state of the period directly. */
    const fastPeriod = async () => {
      const out = plaza.step(controllerRef.current);
      out.lanes.forEach((lane, i) => {
        const ids = laneCarsRef.current[i].slice();
        if (lane.entered) ids.push(newCar(0, 0, geo.carW, geo.carH));
        const staying = ids.filter((_, k) => lane.departures[k] === 0);
        drop(ids.filter((_, k) => lane.departures[k] > 0));
        laneCarsRef.current[i] = staying;
        staying.forEach((id, slot) => move(id, { ...geo.slotPos(i, slot), dur: 0, opacity: 1 }));
      });
      render();
      setGates(new Array(N).fill(false));
      setBubbles(null);
      setMissed(out.lanes.map((l) => l.missed));
      setAtts(attendantsFor(out, 0));
      setHud((h) => ({ ...h, t: out.t, rushHour: out.rushHour, banner: '' }));
      await endOfPeriod(out);
      await sleep(16);
    };

    const animatedPeriod = async () => {
      const out = plaza.step(controllerRef.current);
      setHud((h) => ({ ...h, t: out.t, rushHour: out.rushHour, banner: '' }));
      setMissed(null);
      // 1. the platoon drives in
      const platoon: number[] = [];
      for (let k = 0; k < out.capA; k++) platoon.push(newCar(-60, geo.platoonPos(k).y, geo.pW, geo.pH));
      render();
      await painted();
      if (!alive()) return;
      platoon.forEach((id, k) => move(id, { ...geo.platoonPos(k), dur: d(550) }));
      render();
      await sleep(d(580));
      if (!alive()) return;
      // 2. what each booth asks for
      if (showRequests) {
        setBubbles(out.lanes.map((l) => ({ a: l.aReq, b: l.bReq })));
        await sleep(d(700));
        if (!alive()) return;
        setBubbles(null);
      }
      // 3. the lotteries: attendants walk to the winners, the winners' gates lift
      setAtts(attendantsFor(out, d(500)));
      setGates(out.lanes.map((l) => l.a === 1));
      await sleep(d(520));
      if (!alive()) return;
      // 4. cars drive in; the others turn red and drive away
      let next = 0;
      const lost: number[] = [];
      out.lanes.forEach((lane, i) => {
        if (!lane.a) return;
        const id = platoon[next++];
        if (id === undefined) return;
        if (lane.entered) {
          const slot = laneCarsRef.current[i].length;
          laneCarsRef.current[i].push(id);
          move(id, { ...geo.slotPos(i, slot), w: geo.carW, h: geo.carH, dur: d(720) });
        } else {
          move(id, { x: GATE_X - 24, y: geo.laneCy(i), lost: true, opacity: 0, dur: d(720) });
          lost.push(id);
        }
      });
      for (; next < platoon.length; next++) {
        const id = platoon[next];
        move(id, { x: -60, lost: true, opacity: 0, dur: d(720) });
        lost.push(id);
      }
      setMissed(out.lanes.map((l) => l.missed));
      render();
      await sleep(d(740));
      if (!alive()) return;
      drop(lost);
      setGates(new Array(N).fill(false));
      // 5. cars pay and leave through the booth, each lane on its own clock
      const buckets = 6;
      for (let bkt = 1; bkt <= buckets; bkt++) {
        let moved = false;
        out.lanes.forEach((lane, i) => {
          laneCarsRef.current[i].forEach((id, k) => {
            const dep = lane.departures[k];
            if (dep > 0 && Math.ceil((dep / lane.tau) * buckets) === bkt) {
              move(id, { x: geo.exitX, opacity: 0, dur: d(520) });
              moved = true;
            }
          });
        });
        if (moved) render();
        await sleep(d(1150) / buckets);
        if (!alive()) return;
      }
      out.lanes.forEach((lane, i) => {
        const ids = laneCarsRef.current[i];
        drop(ids.filter((_, k) => lane.departures[k] > 0));
        laneCarsRef.current[i] = ids.filter((_, k) => lane.departures[k] === 0);
        laneCarsRef.current[i].forEach((id, slot) => move(id, { ...geo.slotPos(i, slot), dur: d(380) }));
      });
      render();
      await sleep(d(400));
      if (!alive()) return;
      await endOfPeriod(out);
    };

    (async () => {
      while (alive()) {
        if (!playingRef.current || !visibleRef.current || document.hidden) {
          await sleep(250);
          continue;
        }
        if (turbo()) await fastPeriod();
        else await animatedPeriod();
      }
    })();
    return () => {
      genRef.current++;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, N, rule, seed, resetKey, showRequests, geo]);

  // The static part of the picture (asphalt, booths' bodies, labels) only changes with the geometry.
  const background = useMemo(
    () => (
      <g>
        <rect x={0} y={0} width={W} height={H} fill="#e8eee4" rx={14} />
        <rect x={0} y={TOP - 8} width={GATE_X - 10} height={H - TOP - BOTTOM + 14} fill="#d9e0d3" rx={10} />
        <text x={18} y={TOP + 4} className="svg-label">platoon: {capA} cars</text>
        <rect x={560} y={4} width={430} height={52} rx={10} fill="#dfe9f3" />
        <text x={568} y={17} className="svg-label">attendants' room: {capB}</text>
        {Array.from({ length: N }, (_, i) => (
          <LaneStrip key={i} x0={GATE_X} x1={LANE_END} cy={geo.laneCy(i)} h={geo.asphalt} slots={K} />
        ))}
      </g>
    ),
    [N, K, capA, capB, geo],
  );

  const gateDur = speed >= TURBO ? 0 : 280 / speed;
  const idle = !playing && Object.keys(cars).length === 0;
  return (
    <div className="plaza-stage" style={{ height }} ref={boxRef}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio={`xMidYMid ${fit}`} role="img" aria-label={title ?? 'Animated toll plaza'}>
        {background}
        {Array.from({ length: N }, (_, i) => (
          <Booth key={i} x={geo.boothX} y={geo.laneCy(i) - geo.booth / 2} size={geo.booth} missed={missed?.[i] ?? false} />
        ))}
        {gates.map((open, i) => (
          <Gate key={i} x={GATE_X - 3} y={geo.laneCy(i) + geo.asphalt / 2} len={geo.asphalt} open={open} dur={gateDur} />
        ))}
        {idle &&
          Array.from({ length: capA }, (_, k) => {
            const pos = geo.platoonPos(k);
            return (
              <g key={`idle-${k}`} transform={`translate(${pos.x},${pos.y})`}>
                <Car w={geo.pW} h={geo.pH} color={CAR_COLORS[(k + 1) % CAR_COLORS.length]} />
              </g>
            );
          })}
        {Object.values(cars).map((c) => (
          <CarSprite key={c.id} x={c.x} y={c.y} dur={c.dur} opacity={c.opacity} w={c.w} h={c.h} color={c.color} lost={c.lost} />
        ))}
        {atts.map((a) => (
          <AttendantSprite
            key={a.id} x={a.x} y={a.y} dur={a.dur} opacity={a.opacity}
            size={geo.att} shared={a.shared}
          />
        ))}
        {bubbles?.map((b, i) => (
          <RequestBubble key={i} x={geo.bubbleX} y={geo.laneCy(i)} h={geo.bubbleH} car={b.a === 1} attendant={b.b === 1} />
        ))}
        <text x={W - 14} y={H - 8} textAnchor="end" className="svg-label">
          rush hour {hud.rushHour + 1} · period t = {hud.t} (of 0–{model.T - 1})
        </text>
        {hud.banner && (
          <g transform={`translate(${(GATE_X + LANE_END) / 2}, ${H / 2})`}>
            <rect x={-150} y={-24} width={300} height={48} rx={12} fill="rgba(255,255,255,0.94)" stroke="#999" />
            <text textAnchor="middle" y={7} fontSize={20} fontWeight={700} fill="#333">{hud.banner}</text>
          </g>
        )}
      </svg>
    </div>
  );
}
