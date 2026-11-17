import { useEffect, useState } from 'react';
import { Hero } from './sections/Hero';
import { SectionActions, SectionCosts, SectionState, SectionTime, SectionTransitions } from './sections/booth';
import { SectionBellman } from './sections/planning';
import { SectionManyBooths, SectionSelfish } from './sections/plaza';
import { SectionLearning, SectionPlan, SectionPrices, SectionRounding } from './sections/coordinator';
import { SectionVerdict } from './sections/results';
import { Playground } from './sections/Playground';

const PARTS = [
  { id: 'booth', label: 'One booth', group: 'A booth alone' },
  { id: 'time', label: 'Time', group: 'A booth alone' },
  { id: 'actions', label: 'Decisions', group: 'A booth alone' },
  { id: 'transitions', label: 'Luck', group: 'A booth alone' },
  { id: 'costs', label: 'Cost', group: 'A booth alone' },
  { id: 'bellman', label: 'Plan backwards', group: 'A booth alone' },
  { id: 'many', label: 'Many booths', group: 'Sharing' },
  { id: 'selfish', label: 'Selfish', group: 'Sharing' },
  { id: 'coordinator', label: 'AI plan', group: 'The AI coordinator' },
  { id: 'prices', label: 'Prices', group: 'The AI coordinator' },
  { id: 'rounding', label: 'Orders', group: 'The AI coordinator' },
  { id: 'learning', label: 'Learning', group: 'The AI coordinator' },
  { id: 'verdict', label: 'Verdict', group: 'Verdict' },
  { id: 'playground', label: 'Playground', group: 'Play' },
];

function Nav() {
  const [active, setActive] = useState('');
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    // at most one layout read per frame while scrolling
    let queued = false;
    const update = () => {
      queued = false;
      const h = document.documentElement;
      setProgress(h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight));
      let current = '';
      for (const p of PARTS) {
        const el = document.getElementById(p.id);
        if (el && el.getBoundingClientRect().top < window.innerHeight * 0.4) current = p.id;
      }
      setActive(current);
    };
    const onScroll = () => {
      if (!queued) {
        queued = true;
        requestAnimationFrame(update);
      }
    };
    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);
  return (
    <nav className="topnav" aria-label="Sections">
      <a href="#top" className="brand">Toll Plaza</a>
      <div className="nav-links">
        {PARTS.map((p, i) => (
          <a key={p.id} href={`#${p.id}`} className={p.id === active ? 'on' : ''} title={p.group}>
            {p.id === 'playground' ? '★' : i + 1}
            <span>{p.label}</span>
          </a>
        ))}
      </div>
      <div className="nav-progress" style={{ width: `${progress * 100}%` }} />
    </nav>
  );
}

function PartTitle({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <div className="part-title" id={id}>
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  );
}

export default function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <PartTitle id="part-1" title="Part 1 · A booth on its own">We build the world from scratch: what a booth sees, what it decides, what luck does, and what it costs.</PartTitle>
        <SectionState />
        <SectionTime />
        <SectionActions />
        <SectionTransitions />
        <SectionCosts />
        <SectionBellman />
        <PartTitle id="part-2" title="Part 2 · Sharing">Many booths share the same cars and the same helpers. What happens when each one looks after itself?</PartTitle>
        <SectionManyBooths />
        <SectionSelfish />
        <PartTitle id="part-3" title="Part 3 · The AI coordinator">A central planner that thinks in fractions, sets prices, gives orders, and can compute its plan step by step.</PartTitle>
        <SectionPlan />
        <SectionPrices />
        <SectionRounding />
        <SectionLearning />
        <PartTitle id="part-4" title="Part 4 · The verdict">So, is selfishness really a problem here? The big simulations answer.</PartTitle>
        <SectionVerdict />
        <Playground />
      </main>
      <footer className="footer">
        <p>
          <a href="#top">↑ Back to the top</a> · This page is a high-school companion to the undergraduate tutorial "The Tragedy of the Toll
          Plaza". Every demo runs in your browser with the model of that tutorial; the numbers marked as results come from its simulations.
        </p>
        <p className="muted">The toll plaza is a teaching picture for queues that share limited resources (for example, computer servers that share jobs and power). It is not a model of any real plaza.</p>
      </footer>
    </>
  );
}
