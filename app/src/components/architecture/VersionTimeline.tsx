import { useState } from 'react';
import { versionSteps, versions, type Version } from '../../content/versions';
import { formatFact } from '../../lib/format';
import { SplitTag } from '../ui/SplitTag';

const points = versions.filter((v): v is Version & { leaderboard: NonNullable<Version['leaderboard']> } => !!v.leaderboard);

// Chart geometry in viewBox units.
const W = 640;
const H = 280;
const PAD = { l: 56, r: 24, t: 24, b: 40 };
const Y_MIN = 0.93;
const Y_MAX = 0.99;
const x = (i: number) => PAD.l + (i * (W - PAD.l - PAD.r)) / (points.length - 1);
const y = (v: number) => PAD.t + ((Y_MAX - v) / (Y_MAX - Y_MIN)) * (H - PAD.t - PAD.b);
const ticks = [0.93, 0.95, 0.97, 0.99];

/** Public leaderboard F0.5 by version (§4.5), with the problem and fix behind each step. */
export function VersionTimeline() {
  const [active, setActive] = useState(points.length - 1);
  const v = points[active];
  const step = v ? versionSteps.find((s) => s.to === v.id) : undefined;
  const line = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i)},${y(p.leaderboard.value)}`).join(' ');

  return (
    <div className="mt-12 grid gap-6 lg:grid-cols-[1.5fr_1fr]">
      <figure className="glass rounded-card p-4 md:p-6">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-labelledby="lb-title">
          <title id="lb-title">
            {`Public leaderboard F0.5 by version: ${points.map((p) => `${p.id} ${formatFact(p.leaderboard)}`).join(', ')}`}
          </title>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--color-line)" />
              <text x={PAD.l - 10} y={y(t) + 4} textAnchor="end" className="fill-muted font-mono text-[11px]">
                {t.toFixed(2)}
              </text>
            </g>
          ))}
          <path d={line} fill="none" stroke="var(--color-accent)" strokeWidth="2" strokeLinejoin="round" />
          {points.map((p, i) => (
            <g
              key={p.id}
              role="button"
              tabIndex={0}
              aria-label={`${p.id}: ${formatFact(p.leaderboard)}`}
              aria-pressed={i === active}
              onClick={() => setActive(i)}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), setActive(i))}
              className="cursor-pointer outline-none [&:focus-visible>circle:last-of-type]:stroke-ink"
            >
              <circle cx={x(i)} cy={y(p.leaderboard.value)} r="18" fill="transparent" />
              {i === active && <circle cx={x(i)} cy={y(p.leaderboard.value)} r="11" fill="var(--color-accent)" fillOpacity="0.18" />}
              <circle
                cx={x(i)}
                cy={y(p.leaderboard.value)}
                r="6"
                fill={p.leaderboard.approx ? 'var(--color-canvas)' : 'var(--color-accent)'}
                stroke="var(--color-accent)"
                strokeWidth="2"
                strokeDasharray={p.leaderboard.approx ? '3 2' : undefined}
              />
              <text x={x(i)} y={H - 14} textAnchor="middle" className={`font-mono text-[12px] ${i === active ? 'fill-ink' : 'fill-muted'}`}>
                {p.id}
              </text>
            </g>
          ))}
        </svg>
        <figcaption className="mt-2 text-xs text-muted">
          Public leaderboard F0.5 from our submission history. The v1 value is approximate (hollow point). v3a was not submitted.
        </figcaption>
      </figure>

      {v && (
        <div className="glass rounded-card p-6" aria-live="polite">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-lg text-glow">{v.id}</span>
            {v.final && <span className="text-xs text-match">final submission</span>}
          </div>
          <dl className="mt-4 grid grid-cols-2 gap-3">
            <div>
              <dt className="flex flex-wrap items-center gap-1 text-xs text-muted">
                F0.5 <SplitTag split={v.leaderboard.split} />
              </dt>
              <dd className="mt-1 font-mono text-2xl text-ink">{formatFact(v.leaderboard)}</dd>
            </div>
            <div>
              <dt className="flex flex-wrap items-center gap-1 text-xs text-muted">
                F0.5 <SplitTag split={v.validation.split} />
              </dt>
              <dd className="mt-1 font-mono text-2xl text-ink">{formatFact(v.validation)}</dd>
            </div>
          </dl>
          {v.id === 'v1' && <p className="mt-2 text-xs text-muted">v1 was validated on the raw train distribution.</p>}
          {step && <p className="mt-5 text-sm text-ink/90">{step.text.text}</p>}
          {v.problem && (
            <div className="mt-5 space-y-3 border-t border-line pt-5 text-sm">
              <p>
                <span className="text-unsure">Problem observed: </span>
                <span className="text-ink/90">{v.problem.text}</span>
              </p>
              {v.fix && (
                <p>
                  <span className="text-match">Fix in the next version: </span>
                  <span className="text-ink/90">{v.fix.text}</span>
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
