import { funnel } from '../../content/blocking';
import { decision } from '../../content/decision';
import type { Fact } from '../../content/fact';
import { stageById, type StageId } from '../../content/pipeline';
import { formatFact } from '../../lib/format';
import { Reveal } from '../ui/Reveal';
import { SplitTag } from '../ui/SplitTag';

interface Row {
  label: string;
  note: string;
  /** The pipeline stage this count is measured after. */
  stage: StageId;
  fact: Fact;
  digits?: number;
  also?: { fact: Fact; label: string };
}

const rows: Row[] = [
  { label: 'Retrieved', note: 'Union of sparse keys and the dense bi-encoder', stage: 'union', fact: funnel.union, digits: 1 },
  {
    label: 'After the re-ranker cut',
    note: 'Two-pass LightGBM keeps a short list',
    stage: 'rerankCut',
    fact: funnel.afterCutValidation,
    also: { fact: funnel.afterCutTest, label: 'on the test set' },
  },
  { label: 'Matched', note: `Exclusive assignment, p ≥ ${formatFact(decision.threshold)}`, stage: 'assign', fact: funnel.matchesTest },
];

/** Candidates per business at each step. Bars are on a linear scale against the union. */
export function CandidateFunnel() {
  const max = funnel.union.value;
  return (
    <div className="mt-12 grid gap-4">
      {rows.map((row, i) => (
        <Reveal key={row.label} delay={i * 0.1} className="glass rounded-card p-5 md:p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
            <div>
              <p className="font-medium text-ink">
                {row.label} <span className="ml-1 font-mono text-xs font-normal text-glow">after stage {stageById(row.stage).order}</span>
              </p>
              <p className="text-sm text-muted">{row.note}</p>
            </div>
            <p className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-3xl text-ink">{formatFact(row.fact, { digits: row.digits })}</span>
              <span className="text-sm text-muted">per business</span>
              <SplitTag split={row.fact.split} />
            </p>
          </div>
          <div className="mt-4 h-3 overflow-hidden rounded-full bg-canvas">
            <div
              className="h-full origin-left rounded-full bg-accent"
              style={{ width: `${Math.max((row.fact.value / max) * 100, 1.5)}%` }}
            />
          </div>
          {row.also && (
            <p className="mt-3 flex flex-wrap items-center gap-2 text-sm text-muted">
              <span className="font-mono text-ink">{formatFact(row.also.fact)}</span> {row.also.label}
              <SplitTag split={row.also.fact.split} />
            </p>
          )}
        </Reveal>
      ))}
    </div>
  );
}
