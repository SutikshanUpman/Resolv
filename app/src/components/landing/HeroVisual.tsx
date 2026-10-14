import { useEffect, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { provenanceLabels } from '../../content/variants';
import { easeOutExpo } from '../../lib/motion';
import { useShowcase } from '../../lib/demo/useShowcase';
import type { ShowcaseExample } from '../../lib/demo/types';
import { Chip } from '../ui/Chip';

/** Where each record card starts, as a fraction of half the stage size: streams from the left and right. */
const LANES = [
  { x: -1, y: -0.72 },
  { x: 1, y: -0.5 },
  { x: -1, y: 0.02 },
  { x: 1, y: 0.25 },
  { x: -1, y: 0.76 },
  { x: 1, y: 0.95 },
] as const;

const CYCLE_MS = 6200;
const MERGE_AT_MS = 2600;

/**
 * Real records of one business drift in from both sides, merge into the centre, and become one
 * entity card. Cycles through the landing examples. Decorative: the same data is shown as text
 * in the "Same business" section below.
 */
export function HeroVisual() {
  const showcase = useShowcase();
  const reduced = useReducedMotion();
  const [index, setIndex] = useState(0);
  // Index of the example whose records have merged; the cards merge once per cycle.
  const [mergedIndex, setMergedIndex] = useState<number | null>(null);
  const merged = reduced === true || mergedIndex === index;

  const examples = showcase.status === 'ready' ? showcase.examples : [];
  const example: ShowcaseExample | undefined = examples[index % Math.max(examples.length, 1)];

  useEffect(() => {
    if (examples.length === 0 || reduced) return;
    const merge = window.setTimeout(() => setMergedIndex(index), MERGE_AT_MS);
    const next = window.setTimeout(() => setIndex((i) => i + 1), CYCLE_MS);
    return () => {
      window.clearTimeout(merge);
      window.clearTimeout(next);
    };
  }, [index, examples.length, reduced]);

  return (
    <div
      aria-hidden="true"
      className="relative mx-auto aspect-[1/1.02] w-full max-w-[34rem] overflow-hidden rounded-card border border-line bg-raised/50"
    >
      {/* Faint ruled lines, like a notebook page. */}
      <div className="absolute inset-0 bg-[linear-gradient(var(--color-line)_1px,transparent_1px)] bg-[size:100%_2rem] opacity-60" />

      <div className="absolute inset-0 grid place-items-center">
        <AnimatePresence mode="popLayout">
          {example &&
            example.records.slice(0, LANES.length).map((r, i) => {
              const lane = LANES[i] ?? LANES[0];
              return (
                <motion.div
                  key={`${example.id}-${r.id}`}
                  className="glass absolute w-[46%] rounded-xl px-3 py-2 shadow-card"
                  initial={{ opacity: 0, x: `${lane.x * 135}%`, y: `${lane.y * 230}%`, scale: 0.92 }}
                  animate={
                    merged
                      ? { opacity: 0, x: '0%', y: '0%', scale: 0.55 }
                      : { opacity: 1, x: `${lane.x * 58}%`, y: `${lane.y * 230}%`, scale: 1 }
                  }
                  exit={{ opacity: 0 }}
                  transition={{ duration: merged ? 0.9 : 1.1, ease: easeOutExpo, delay: merged ? i * 0.04 : i * 0.12 }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-[0.6rem] text-muted">{r.source}</span>
                    {r.script && <span className="text-[0.6rem] text-glow">{r.script}</span>}
                  </div>
                  <p className="truncate text-[0.8rem] font-medium text-ink">{r.name}</p>
                  <p className="truncate text-[0.65rem] text-muted">{r.address ?? 'no address'}</p>
                </motion.div>
              );
            })}
        </AnimatePresence>

        <AnimatePresence>
          {example && merged && (
            <motion.div
              key={`entity-${example.id}`}
              className="absolute w-[72%] rounded-card border border-accent/60 bg-surface p-5 shadow-glow"
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ duration: 0.7, ease: easeOutExpo }}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-[0.65rem] text-glow">One business</span>
                <Chip tone={example.provenance === 'verified' ? 'match' : 'accent'}>
                  {provenanceLabels[example.provenance].short}
                </Chip>
              </div>
              <p className="mt-3 text-lg leading-tight font-semibold text-ink">{example.s1.name}</p>
              <p className="mt-1 line-clamp-2 text-xs text-muted">{example.s1.address}</p>
              <p className="mt-4 text-xs text-muted">
                <span className="font-mono text-match">{example.records.length}</span> records from S2 and S3 ·{' '}
                {example.country}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
