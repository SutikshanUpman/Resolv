import { useState } from 'react';
import { AnimatePresence } from 'motion/react';
import { useShowcase } from '../../lib/demo/useShowcase';
import { SectionHeading } from '../ui/SectionHeading';
import { ShowcaseExampleView } from './ShowcaseExampleView';

export function SameBusiness() {
  const showcase = useShowcase();
  const [selected, setSelected] = useState(0);

  return (
    <section className="container-page py-24 md:py-32" aria-labelledby="same-business">
      <SectionHeading
        id="same-business"
        eyebrow="Same business, different spelling"
        title="One business, written many ways"
        lead="These are real records from the challenge data. Pick a business to see how its records in Sources 2 and 3 come together as one."
      />

      {showcase.status === 'loading' && <div className="mt-12 h-96 animate-pulse rounded-card bg-surface/60" />}
      {showcase.status === 'error' && (
        <p className="mt-12 text-danger" role="alert">
          The examples didn’t load ({showcase.message}). Reload the page to try again.
        </p>
      )}
      {showcase.status === 'ready' && (
        <>
          <div role="tablist" aria-label="Examples" className="mt-12 flex gap-2 overflow-x-auto pb-2">
            {showcase.examples.map((ex, i) => (
              <button
                key={ex.id}
                role="tab"
                type="button"
                aria-selected={i === selected}
                onClick={() => setSelected(i)}
                className={`shrink-0 rounded-full border px-4 py-2 text-sm transition-colors ${
                  i === selected ? 'border-accent bg-accent/15 text-ink' : 'border-line text-muted hover:text-ink'
                }`}
              >
                {ex.s1.name}
                <span className="ml-2 text-xs text-muted">{ex.country}</span>
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            {showcase.examples[selected] && (
              <ShowcaseExampleView key={selected} index={selected} example={showcase.examples[selected]} />
            )}
          </AnimatePresence>
        </>
      )}
    </section>
  );
}
