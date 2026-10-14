import { dataset } from '../../content/dataset';
import { headline } from '../../content/metrics';
import type { Fact } from '../../content/fact';
import { formatFact } from '../../lib/format';
import { CountUp } from '../ui/CountUp';
import { SplitTag } from '../ui/SplitTag';

interface Stat {
  fact: Fact;
  label: string;
  digits?: number;
  rounding?: 'down' | 'nearest';
}

const stats: Stat[] = [
  // CLAUDE.md allows the leaderboard 0.985978 to be shown as 0.986; every other number rounds down.
  { fact: headline.leaderboardF05, label: 'Macro F0.5', digits: 3, rounding: 'nearest' },
  { fact: headline.validationPrecision, label: 'Macro precision', digits: 3 },
  { fact: dataset.testS1, label: 'Reference businesses' },
  { fact: dataset.testRecords, label: 'Records searched' },
];

export function StatBand() {
  return (
    <section aria-label="Results in numbers" className="border-y border-line bg-surface/40">
      <dl className="container-page grid grid-cols-2 gap-x-6 gap-y-10 py-12 md:grid-cols-5">
        {stats.map(({ fact, label, digits, rounding }) => {
          const final = formatFact(fact, { digits, rounding });
          const shown = Number(final.replace(/[^0-9.]/g, ''));
          return (
            <div key={label} className="min-w-0">
              <dt className="flex flex-wrap items-center gap-2 text-sm text-muted">
                {label} <SplitTag split={fact.split} />
              </dt>
              <dd className="mt-2 font-mono text-[clamp(1.5rem,1.1rem+1.6vw,2.4rem)] font-medium tracking-tight text-ink tabular-nums">
                <CountUp value={shown} decimals={digits ?? (final.includes('.') ? final.split('.')[1]?.length ?? 0 : 0)} final={final} />
              </dd>
            </div>
          );
        })}
        <div className="min-w-0">
          <dt className="text-sm text-muted">Countries</dt>
          <dd className="mt-2 font-mono text-[clamp(1.5rem,1.1rem+1.6vw,2.4rem)] font-medium tracking-tight text-ink">
            {dataset.countries.length}
          </dd>
          <p className="mt-1 text-xs text-muted">{dataset.countriesText.text}</p>
        </div>
      </dl>
    </section>
  );
}
