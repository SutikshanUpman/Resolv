import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { pipeline, type StageId } from '../content/pipeline';
import { PageHeader } from '../components/layout/PageHeader';
import { PageMeta } from '../components/layout/PageMeta';
import { Button } from '../components/ui/Button';
import { ButtonLink } from '../components/ui/ButtonLink';
import { RunStageRow } from '../components/workspace/RunStageRow';
import { resolver } from '../lib/resolver';
import type { CounterKey, ResolverInput, ResolverRun } from '../lib/resolver/types';
import type { StageStatus } from '../lib/workspace/runLabels';
import { isReady, useWorkspace, workspace, type WorkspaceState } from '../lib/workspace/store';

function inputFrom(s: WorkspaceState): ResolverInput | null {
  if (!isReady(s)) return null;
  const records = (id: 'S1' | 'S2' | 'S3') => {
    const slot = s.slots[id];
    return slot.status === 'ready' ? slot.validation.records : [];
  };
  const upload = Object.values(s.slots).some((slot) => slot.status === 'ready' && slot.origin === 'upload');
  return { origin: upload ? 'upload' : 'sample', s1: records('S1'), s2: records('S2'), s3: records('S3') };
}

export function RunPage() {
  const ws = useWorkspace();
  const navigate = useNavigate();
  const [input] = useState(() => inputFrom(ws));
  const [statuses, setStatuses] = useState<Partial<Record<StageId, StageStatus>>>({});
  const [counters, setCounters] = useState<Partial<Record<CounterKey, number>>>({});
  const [error, setError] = useState<string | null>(null);
  const run = useRef<ResolverRun | null>(null);

  useEffect(() => {
    if (!input) return;
    const current = resolver.resolve(input);
    run.current = current;
    let live = true;
    void (async () => {
      for await (const e of current.events) {
        if (!live) return;
        if (e.type === 'stage-start') setStatuses((s) => ({ ...s, [e.stage]: 'running' }));
        else if (e.type === 'stage-end') setStatuses((s) => ({ ...s, [e.stage]: 'done' }));
        else if (e.type === 'progress') setCounters((c) => ({ ...c, [e.counter]: e.value }));
        else if (e.type === 'error') setError(e.message);
        else if (e.type === 'done') {
          workspace.setResult(e.result);
          void navigate('/app/results');
        }
      }
    })();
    return () => {
      live = false;
      current.cancel();
    };
  }, [input, navigate]);

  if (!input) {
    return (
      <>
        <PageMeta title="Processing" />
        <PageHeader eyebrow="Workspace" title="Nothing to process yet" lead="Add three sources, or load the sample data, then press Refine." />
        <div className="container-page">
          <ButtonLink to="/app">Go to the workspace</ButtonLink>
        </div>
      </>
    );
  }

  const done = pipeline.filter((s) => statuses[s.id] === 'done').length;

  return (
    <>
      <PageMeta title="Processing" />
      <PageHeader
        eyebrow="Demo mode: precomputed results"
        title="Refining your sources"
        lead="Each stage of the pipeline in order. The counts are the real figures from our submission for the sample data."
      />
      <section className="container-page">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface" role="progressbar" aria-valuemin={0} aria-valuemax={pipeline.length} aria-valuenow={done} aria-label="Stages complete">
            <div className="h-full origin-left rounded-full bg-accent transition-transform duration-500" style={{ transform: `scaleX(${done / pipeline.length})` }} />
          </div>
          <Button variant="secondary" onClick={() => run.current?.skip()} disabled={!!error}>
            Skip to results
          </Button>
        </div>

        {error && (
          <div role="alert" className="mt-6 rounded-card border border-danger/40 bg-danger/10 p-6">
            <p className="font-semibold text-ink">The run stopped</p>
            <p className="mt-2 text-sm text-ink/85">{error}</p>
            <ButtonLink to="/app" className="mt-4" variant="secondary">
              Back to the workspace
            </ButtonLink>
          </div>
        )}

        <ol className="mt-8 grid gap-2">
          {pipeline.map((stage) => (
            <RunStageRow key={stage.id} stage={stage} status={statuses[stage.id] ?? 'waiting'} counters={counters} />
          ))}
        </ol>
      </section>
    </>
  );
}
