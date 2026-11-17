# The Tragedy of the Toll Plaza: interactive web page

This is a single-page, scroll-driven introduction to Markov decision processes for high-school students. It is the
companion of the undergraduate tutorial `papers/Tutorial/main.tex`.

As the reader scrolls, the text arrives one step at a time and the illustration next to it reacts. The story builds
one booth (state, time, decisions, luck, cost, planning backwards). It then shares cars and attendants among many
booths (a weakly coupled MDP), lets every booth act selfishly, and introduces the AI coordinator (thinking in
fractions, prices, orders, learning). It ends with the verdict and a playground where everything runs at once.

The toll plaza is a teaching picture for queues that share limited resources. It is not a model of any real plaza.

## Run

Use Node 20 or later.

```bash
npm --prefix web install
npm --prefix web run dev      # http://localhost:5173
npm --prefix web test         # engine tests against the Python reference values
npm --prefix web run build    # static site in web/dist (relative paths; open from any folder or static host)
```

## What runs in the browser

The demos compute everything live with a TypeScript port of the model, in `src/engine`.

| File | What it does |
|---|---|
| `model.ts` | transition kernel (concurrent service, geometric period length), costs |
| `dp.ts` | backward induction for the selfish booths and the price-taking booths |
| `fluid.ts` | large-N evaluation under rationing (lottery) or congestion (shared attendants) |
| `gda.ts` | gradient descent-ascent learning of the coordinator's plan (run in a web worker, `src/workers/gda.worker.ts`) |
| `sim.ts` | the N-booth plaza: requests, lotteries, the coordinator's rounding and repair, and per-lane clocks and departures |

`src/engine/engine.test.ts` checks this engine against `src/data/reference.json`, which comes from the Python
implementation in `lp_learner`. It checks:

- the kernels;
- backward induction (both tie-breaks) and the price-taking policy;
- the LP's induced policy;
- the large-N costs, masses and grants under every rule;
- the GDA iterates from a zero start (agreement to 1e-8);
- the simulator: the kernel statistically, rounding and repair, and that the coordinator never exceeds a budget.

## Data (generated, do not edit)

`src/data/model.json` and `src/data/reference.json` are written by the `web` stage (`configs/web.toml`,
`lp_learner/visualizations/web_export.py`):

```bash
uv run lpl-experiments run web --config configs/web.toml
uv run lpl-experiments summarize web --config configs/web.toml
```

`scripts/reproduce_tutorial.sh --web` runs the whole chain.

`model.json` holds:

- the exact LP plan, its prices and its value for every budget pair of the playground's grid (alpha in
  {0.5, 0.625, 0.75, 0.875, 1}, beta in {0.1, 0.25, 0.4, 0.5, 0.75, 1});
- the recorded results that the page quotes as "results from our big simulations". These are the tutorial's numbers:
  1000 booths, 2000 rush hours per setting, from `results/commons/summary_tutorial` and `results/finite_n/summary`.

The page keeps live demos and recorded results apart:

- **Live demos** use small plazas (4 to 30 booths) and are noisy.
- **"Many booths" values** are the large-N limit, computed live.
- **"Big simulations"** are the recorded results.

## Layout

| Path | Contents |
|---|---|
| `src/sections/` | the parts of the story: `booth.tsx` (sections 1-5), `planning.tsx` (6), `plaza.tsx` (7-8), `coordinator.tsx` (9-12), `results.tsx` (13), `Playground.tsx`, `Hero.tsx` |
| `src/components/` | the scroll framework (`Scrolly.tsx`), the animated plaza (`PlazaStage.tsx`), the single lane (`LaneView.tsx`), the pie plot of the research paper (`PiePlot.tsx`), charts, controls, KaTeX math (`TeX.tsx`) |
| `src/hooks/useGda.ts` | the learning worker's lifecycle |

## Animations and performance

- **Moving things are memoised SVG sprites** (`CarSprite`, `AttendantSprite` in `src/components/art.tsx`) that glide
  with CSS transitions, so a phase re-renders only the sprites that move. A new sprite is drawn at its start
  position, and the browser paints it before it moves (`afterPaint`, with a timer fallback for throttled tabs).
- **Single lane (`LaneView.tsx`, `useLane`).** An arriving car lifts the gate, drives to the end of the queue, and the
  gate closes again. A paying car drives out through the booth while the queue moves up. A car that cannot get in
  turns red and fades.
- **Plaza (`PlazaStage.tsx`).** Each period plays in order:
  1. the platoon drives in;
  2. request bubbles show what each booth wants;
  3. the lotteries run: attendants walk to the winning booths and those gates lift;
  4. cars drive into their lanes, and the gates close; a booth without a new car shows a red window;
  5. cars leave on each lane's own clock, and the queues move up.

  Booth, car and attendant sizes scale with the lane height.
- **Turbo** (speed >= 12) skips the phases and draws each period once.
- **Pausing.** Plazas pause while off screen or in a hidden tab.
- **Other costs.** The navigation bar reads the layout at most once per frame. The learning runs in a web worker.
