import type { PipelineStage } from '../../content/pipeline';
import type { CounterKey } from '../../lib/resolver/types';
import { counterLabels, stageCounters, stageNotes, type StageStatus } from '../../lib/workspace/runLabels';

interface RunStageRowProps {
  stage: PipelineStage;
  status: StageStatus;
  counters: Partial<Record<CounterKey, number>>;
}

export function RunStageRow({ stage, status, counters }: RunStageRowProps) {
  const keys = stageCounters[stage.id] ?? [];
  const note = stageNotes[stage.id];
  return (
    <li
      className={`flex flex-col gap-2 rounded-xl border p-4 transition-[opacity,border-color] duration-300 sm:flex-row sm:items-center sm:gap-5 ${
        status === 'running' ? 'border-accent/60 bg-surface shadow-glow' : status === 'done' ? 'border-line bg-surface/60' : 'border-line opacity-50'
      }`}
    >
      <div className="flex min-w-0 flex-1 items-center gap-4">
        <span
          className={`grid size-9 shrink-0 place-items-center rounded-full border font-mono text-xs ${
            status === 'done' ? 'border-match/50 text-match' : status === 'running' ? 'border-accent bg-accent text-white' : 'border-line text-muted'
          }`}
          aria-hidden="true"
        >
          {status === 'done' ? '✓' : stage.order}
        </span>
        <div className="min-w-0">
          <p className="font-medium text-ink">{stage.title}</p>
          {note && <p className="text-xs text-muted">{note}</p>}
        </div>
      </div>
      <dl className="flex flex-wrap gap-x-5 gap-y-1 pl-13 sm:pl-0">
        {keys.map((k) => (
          <div key={k} className="flex items-baseline gap-1.5">
            <dd className="font-mono text-lg text-ink tabular-nums">{(counters[k] ?? 0).toLocaleString('en-US')}</dd>
            <dt className="text-xs text-muted">{counterLabels[k]}</dt>
          </div>
        ))}
      </dl>
      <span className="sr-only">{status === 'done' ? 'done' : status === 'running' ? 'running' : 'waiting'}</span>
    </li>
  );
}
