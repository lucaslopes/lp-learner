import { useEffect, useRef, useState, type ReactNode } from 'react';

export interface Step {
  key: string;
  body: ReactNode;
}

/**
 * A scroll-driven section: the text arrives one step at a time on the left while the illustration on the right stays
 * in view and reacts to the active step. `render(step)` draws the illustration for the step that is in the middle of
 * the screen.
 */
export function Scrolly({ id, number, title, kicker, steps, render, wide = false }: {
  id: string;
  number?: string;
  title: string;
  kicker?: string;
  steps: Step[];
  render: (step: number) => ReactNode;
  wide?: boolean;
}) {
  const [active, setActive] = useState(0);
  const refs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const i = Number((entry.target as HTMLElement).dataset.step);
            setActive(i);
          }
        }
      },
      // On narrow screens the illustration is pinned over the top half, so a step counts as active in the lower part.
      { rootMargin: window.matchMedia('(max-width: 900px)').matches ? '-72% 0px -22% 0px' : '-45% 0px -50% 0px', threshold: 0 },
    );
    refs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, [steps.length]);

  return (
    <section id={id} className={`scrolly ${wide ? 'scrolly-wide' : ''}`}>
      <header className="section-head">
        {number && <span className="section-number">{number}</span>}
        <h2>{title}</h2>
        {kicker && <p className="kicker">{kicker}</p>}
      </header>
      <div className="scrolly-body">
        <div className="scrolly-steps">
          {steps.map((step, i) => (
            <div
              key={step.key}
              ref={(el) => (refs.current[i] = el)}
              data-step={i}
              className={`step ${i === active ? 'is-active' : i < active ? 'is-past' : ''}`}
            >
              <div className="step-card">{step.body}</div>
            </div>
          ))}
          <div className="step-spacer" />
        </div>
        <div className="scrolly-graphic">
          <div className="graphic-sticky">{render(active)}</div>
        </div>
      </div>
    </section>
  );
}
