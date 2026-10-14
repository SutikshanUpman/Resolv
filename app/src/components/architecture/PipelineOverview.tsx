import { overview, type OverviewPhase } from '../../content/overview';
import { stageById } from '../../content/pipeline';
import { formatFact } from '../../lib/format';
import { Reveal } from '../ui/Reveal';
import { SplitTag } from '../ui/SplitTag';

/** Arrow between blocks: points down when stacked, right when the phases sit in a row. */
function FlowArrow({ sideways = false }: { sideways?: boolean }) {
  return (
    <div aria-hidden="true" className={`grid place-items-center text-glow ${sideways ? 'py-1 lg:px-1 lg:py-0' : 'py-1'}`}>
      <svg
        viewBox="0 0 20 20"
        className={`size-5 ${sideways ? 'lg:-rotate-90' : ''}`}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
      >
        <path d="M10 3v13M5 11l5 5 5-5" />
      </svg>
    </div>
  );
}

function stageRange(phase: OverviewPhase): string {
  const orders = phase.stages.map((id) => stageById(id).order);
  const first = Math.min(...orders);
  const last = Math.max(...orders);
  return first === last ? `Stage ${first}` : `Stages ${first}–${last}`;
}

function PhaseCard({ phase }: { phase: OverviewPhase }) {
  return (
    <article className="glass flex h-full flex-col rounded-card p-5 md:p-6">
      <p className="font-mono text-xs text-glow">{stageRange(phase)}</p>
      <h3 className="mt-2 text-h3 font-semibold text-ink">{phase.title}</h3>
      <p className="mt-3 text-sm text-muted">{phase.summary}</p>

      <ol className="mt-5 grid gap-1.5 text-sm">
        {phase.steps.map((step, i) => (
          <li key={step.label}>
            {i > 0 && <span aria-hidden="true" className="mx-auto mb-1.5 block h-2.5 w-px bg-line-strong" />}
            {step.parallel ? (
              <div className="rounded-xl border border-line bg-canvas/60 p-2">
                <p className="px-1 text-xs text-muted">{step.label}, side by side:</p>
                <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                  {step.parallel.map((p) => (
                    <span key={p} className="rounded-lg border border-accent/30 bg-accent/10 px-2 py-1.5 text-center text-xs text-ink">
                      {p}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <p className="rounded-xl border border-line bg-canvas/60 px-3 py-2 text-ink/90">{step.label}</p>
            )}
          </li>
        ))}
      </ol>

      <div className="mt-auto pt-5">
        <div className="border-t border-line pt-4">
          <p className="text-xs tracking-[0.14em] text-muted uppercase">Out of this phase</p>
          {phase.result.fact ? (
            <p className="mt-2 flex flex-wrap items-center gap-2">
              <span className="font-mono text-2xl text-ink">{formatFact(phase.result.fact)}</span>
              <span className="text-sm text-muted">{phase.result.label}</span>
              <SplitTag split={phase.result.fact.split} />
            </p>
          ) : (
            <p className="mt-2 text-ink">{phase.result.label}</p>
          )}
          {phase.result.also && (
            <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted">
              <span className="font-mono text-ink">{formatFact(phase.result.also)}</span> on the test set
              <SplitTag split={phase.result.also.split} />
            </p>
          )}
        </div>
      </div>
    </article>
  );
}

/** The pipeline in three phases, before the ten-stage detail. */
export function PipelineOverview() {
  const { input, output, phases } = overview;
  return (
    <div className="mt-12">
      <Reveal className="glass flex flex-col gap-4 rounded-card p-5 md:flex-row md:items-center md:gap-8 md:p-6">
        <p className="text-xs font-medium tracking-[0.14em] text-glow uppercase md:w-12">{input.title}</p>
        <div className="grid flex-1 gap-4 sm:grid-cols-2">
          <div>
            <p className="text-ink">{input.s1}</p>
            <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted">
              <span className="font-mono text-ink">{formatFact(input.s1Count)}</span> businesses
              <SplitTag split={input.s1Count.split} />
            </p>
          </div>
          <div>
            <p className="text-ink">{input.sources}</p>
            <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted">
              <span className="font-mono text-ink">{formatFact(input.recordCount)}</span> records
              <SplitTag split={input.recordCount.split} />
            </p>
          </div>
        </div>
      </Reveal>

      <FlowArrow />

      <div className="grid lg:grid-cols-[1fr_auto_1fr_auto_1fr] lg:items-stretch">
        {phases.map((phase, i) => (
          <div key={phase.id} className="contents">
            {i > 0 && <FlowArrow sideways />}
            <Reveal delay={i * 0.1} className="h-full min-w-0">
              <PhaseCard phase={phase} />
            </Reveal>
          </div>
        ))}
      </div>

      <FlowArrow />

      <Reveal className="glass flex flex-col gap-4 rounded-card p-5 md:flex-row md:items-center md:gap-8 md:p-6">
        <p className="text-xs font-medium tracking-[0.14em] text-match uppercase md:w-12">{output.title}</p>
        <p className="flex-1 text-ink">{output.rule.text}</p>
      </Reveal>
    </div>
  );
}
