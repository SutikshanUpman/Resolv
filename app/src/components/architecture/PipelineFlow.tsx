import { useState } from 'react';
import { overview, stageShortTitles } from '../../content/overview';
import { pipeline, stageById, type StageId } from '../../content/pipeline';
import { StagePanel } from './StagePanel';

/**
 * The ten stages of §2.2 as a route map: three phase segments of numbered stations on one line.
 * Selecting a station shows its detail card below, and Previous/Next walk the route in order.
 */
export function PipelineFlow() {
  const [selected, setSelected] = useState<StageId>(pipeline[0]?.id ?? 'normalise');
  const current = stageById(selected);
  const index = pipeline.findIndex((s) => s.id === selected);
  const prev = pipeline[index - 1];
  const next = pipeline[index + 1];
  const phase = overview.phases.find((p) => (p.stages as readonly StageId[]).includes(selected));

  return (
    <div className="mt-12">
      <div className="grid gap-6 rounded-card border border-line bg-surface p-5 shadow-card md:grid-cols-[4fr_5fr_1.5fr] md:gap-0 md:p-6">
        {overview.phases.map((p, pi) => (
          <section key={p.id} aria-label={p.title} className={`min-w-0 ${pi > 0 ? 'md:border-l md:border-dashed md:border-line-strong md:pl-4' : ''} md:pr-4`}>
            <p className="text-xs font-medium tracking-[0.12em] text-muted uppercase">
              <span className="font-mono text-glow">{String(pi + 1).padStart(2, '0')}</span> {p.title}
            </p>
            <div className="relative mt-4">
            {/* The line the stations sit on, from the first station's centre to the last's. */}
            <span aria-hidden="true" className="absolute top-5 right-[calc(50%/var(--n))] left-[calc(50%/var(--n))] h-0.5 bg-line-strong" style={{ ['--n' as string]: p.stages.length }} />
            <ol className="relative grid auto-cols-fr grid-flow-col gap-1">
              {p.stages.map((id) => {
                const stage = stageById(id);
                const isSelected = id === selected;
                const passed = stage.order < current.order;
                return (
                  <li key={id} className="relative flex justify-center">
                    <button
                      type="button"
                      aria-expanded={isSelected}
                      aria-controls={`stage-${id}`}
                      onClick={() => setSelected(id)}
                      className="group flex w-full min-w-0 flex-col items-center gap-2 rounded-control px-0.5 pb-1 text-center"
                    >
                      <span
                        className={`relative grid size-10 place-items-center rounded-full border-2 font-mono text-sm transition-colors ${
                          isSelected
                            ? 'border-accent bg-accent text-white'
                            : passed
                              ? 'border-ink bg-ink text-canvas'
                              : 'border-line-strong bg-surface text-ink group-hover:border-ink'
                        }`}
                      >
                        {stage.order}
                      </span>
                      <span className={`text-[0.7rem] leading-tight break-words md:text-xs ${isSelected ? 'font-semibold text-ink' : 'text-muted group-hover:text-ink'}`}>
                        {stageShortTitles[id]}
                      </span>
                      {stage.output && <span className="hidden font-mono text-[0.6rem] text-match lg:block">{stage.output}</span>}
                    </button>
                  </li>
                );
              })}
            </ol>
            </div>
          </section>
        ))}
      </div>

      <StagePanel
        id={`stage-${current.id}`}
        stage={current}
        position={`Stage ${current.order} of ${pipeline.length}${phase ? ` · ${phase.title}` : ''}`}
        prev={prev ? { label: stageShortTitles[prev.id], onSelect: () => setSelected(prev.id) } : undefined}
        next={next ? { label: stageShortTitles[next.id], onSelect: () => setSelected(next.id) } : undefined}
      />
    </div>
  );
}
