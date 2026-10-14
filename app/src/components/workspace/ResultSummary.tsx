import type { ResolveSummary } from '../../lib/resolver/types';

export function ResultSummary({ summary }: { summary: ResolveSummary }) {
  const stats = [
    { label: 'Businesses resolved', value: summary.businesses.toLocaleString('en-US') },
    { label: 'Matches found', value: summary.matches.toLocaleString('en-US') },
    { label: 'Average matches per business', value: summary.avgMatches.toFixed(2) },
    { label: 'Left unmatched', value: `${(summary.unmatchedShare * 100).toFixed(1)}%` },
  ];
  return (
    <dl className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {stats.map((s) => (
        <div key={s.label} className="glass rounded-card p-5">
          <dt className="text-sm text-muted">{s.label}</dt>
          <dd className="mt-2 font-mono text-2xl text-ink tabular-nums md:text-3xl">{s.value}</dd>
        </div>
      ))}
    </dl>
  );
}
