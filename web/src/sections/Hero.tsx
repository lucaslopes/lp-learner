import { useMemo } from 'react';
import { PlazaStage } from '../components/PlazaStage';
import { scenario } from '../engine/data';
import { capacities } from '../engine/model';
import { tableController } from '../engine/sim';

const sc = scenario();

export function Hero() {
  const ctrl = useMemo(() => tableController('selfish', sc.selfish.u), []);
  return (
    <section className="hero" id="top">
      <div className="hero-text">
        <p className="eyebrow">An interactive story about decisions under uncertainty</p>
        <h1>The Tragedy of the Toll Plaza</h1>
        <p className="lede">
          A thousand toll booths, a few helpers, and one smart coordinator. Scroll down to build, piece by piece, the math that
          computers use to plan when luck is involved: <b>Markov decision processes</b>.
        </p>
        <div className="hero-buttons">
          <a className="btn btn-primary" href="#booth">Start the story ↓</a>
          <a className="btn" href="#playground">Jump to the playground</a>
        </div>
        <p className="hero-note">The toll plaza is a picture to help you think. The same math is used for computer servers sharing jobs and power.</p>
      </div>
      <div className="hero-plaza" aria-label="A live toll plaza where every booth looks after itself">
        <PlazaStage model={sc.model} N={10} rule="rationing" controller={ctrl} playing speed={0.9} seed={3} showRequests height="auto" />
        <p className="hero-caption">Live: 10 booths, {capacities(10, sc.params.alpha, sc.params.beta)[0]} cars per platoon, {capacities(10, sc.params.alpha, sc.params.beta)[1]} attendants. Almost every booth asks for both. Who gets what?</p>
      </div>
    </section>
  );
}
