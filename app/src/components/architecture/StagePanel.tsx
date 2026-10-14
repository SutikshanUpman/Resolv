import { motion } from 'motion/react';
import { Link } from 'react-router';
import { models } from '../../content/models';
import { stageOneLiners } from '../../content/overview';
import type { PipelineStage } from '../../content/pipeline';
import { formatFact } from '../../lib/format';
import { sectionAnchor } from '../../lib/docs';
import { Chip } from '../ui/Chip';
import { SplitTag } from '../ui/SplitTag';

interface StepLink {
  label: string;
  onSelect: () => void;
}

interface StagePanelProps {
  id: string;
  stage: PipelineStage;
  /** e.g. "Stage 4 of 10 · Find candidates" */
  position: string;
  prev?: StepLink;
  next?: StepLink;
}

const stepButton =
  'inline-flex items-center gap-2 rounded-control border border-line px-3 py-2 text-sm text-ink transition-colors hover:border-ink disabled:opacity-0';

/** The detail card for the selected pipeline stage. */
export function StagePanel({ id, stage, position, prev, next }: StagePanelProps) {
  return (
    <section id={id} aria-live="polite" aria-label={`Stage ${stage.order}: ${stage.title}`} className="mt-4 rounded-card border border-line bg-surface shadow-card">
      <motion.div
        key={stage.id}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="grid gap-8 p-5 md:grid-cols-[1.4fr_1fr] md:p-8"
      >
        <div className="min-w-0">
          <p className="font-mono text-xs text-glow">{position}</p>
          <h3 className="mt-2 font-serif text-h2 font-semibold text-ink">{stage.title}</h3>
          <p className="mt-3 text-lead text-ink">{stageOneLiners[stage.id]}</p>
          <p className="mt-4 text-muted">{stage.summary}</p>
          {stage.models.length > 0 && (
            <ul className="mt-5 space-y-2">
              {stage.models.map((m) => {
                const model = models[m];
                return (
                  <li key={m} className="flex flex-wrap items-center gap-2 text-sm">
                    <span className="font-mono text-ink">{model.name.text}</span>
                    {model.params && <Chip>{formatFact(model.params)} parameters</Chip>}
                    <Chip>{model.licence.text} licence</Chip>
                  </li>
                );
              })}
            </ul>
          )}
          <p className="mt-5 border-l-2 border-line-strong pl-3 text-xs text-muted">
            In the write-up’s diagram: <span className="font-mono break-words text-ink/80">{stage.diagram.text}</span>
          </p>
          <p className="mt-4 text-sm text-muted">
            Read more:{' '}
            {stage.sections.map((s, i) => (
              <span key={s}>
                {i > 0 && ', '}
                <Link to={`/architecture/docs#${sectionAnchor(s)}`} className="text-glow underline underline-offset-4">
                  §{s}
                </Link>
              </span>
            ))}
            {stage.output && (
              <>
                {' '}
                · writes <span className="font-mono text-match">{stage.output}</span>
              </>
            )}
          </p>
        </div>
        {stage.metrics.length > 0 && (
          <dl className="grid content-start gap-3">
            {stage.metrics.map((m) => (
              <div key={`${m.label}-${m.fact.quote}`} className="border-b border-line pb-3 last:border-b-0">
                <dt className="flex flex-wrap items-center gap-2 text-xs text-muted">
                  {m.label} <SplitTag split={m.fact.split} />
                </dt>
                <dd className="mt-1 font-mono text-2xl text-ink">{formatFact(m.fact)}</dd>
              </div>
            ))}
          </dl>
        )}
      </motion.div>
      <div className="flex items-center justify-between gap-3 border-t border-line px-5 py-3 md:px-8">
        <button type="button" className={stepButton} onClick={prev?.onSelect} disabled={!prev} aria-label={prev ? `Previous stage: ${prev.label}` : undefined}>
          <span aria-hidden="true">←</span> {prev?.label ?? ''}
        </button>
        <button type="button" className={stepButton} onClick={next?.onSelect} disabled={!next} aria-label={next ? `Next stage: ${next.label}` : undefined}>
          {next?.label ?? ''} <span aria-hidden="true">→</span>
        </button>
      </div>
    </section>
  );
}
