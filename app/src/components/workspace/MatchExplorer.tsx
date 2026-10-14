import { useMemo, useState } from 'react';
import type { BusinessRecord, ResolveResult } from '../../lib/resolver/types';
import { Chip } from '../ui/Chip';

const LIST_LIMIT = 60;

function RecordCard({ record, matched }: { record: BusinessRecord; matched: boolean }) {
  return (
    <li className={`rounded-xl border p-3 ${matched ? 'border-match/30 bg-surface/70' : 'border-line bg-canvas/40'}`}>
      <div className="flex flex-wrap items-center gap-1.5">
        <Chip tone={matched ? 'match' : 'neutral'}>{matched ? 'Matched' : 'Considered'}</Chip>
        <Chip>{record.source}</Chip>
        <Chip>{record.country}</Chip>
        <span className="ml-auto font-mono text-[0.65rem] text-muted">{record.id}</span>
      </div>
      <p className="mt-2 font-medium break-words text-ink">{record.name}</p>
      <p className="mt-0.5 text-sm break-words text-muted">{record.address ?? 'No address'}</p>
    </li>
  );
}

/** Pick a business; see its S2 and S3 records side by side, matched first, then candidates it was not matched to. */
export function MatchExplorer({ result }: { result: ResolveResult }) {
  const businesses = useMemo(() => [...result.records.values()].filter((r) => r.source === 'S1'), [result]);
  const countries = useMemo(() => [...new Set(businesses.map((b) => b.country))].sort(), [businesses]);
  const [country, setCountry] = useState('all');
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const matchesBy = useMemo(() => new Map(result.matches.map((r) => [r.s1Id, r.ids])), [result]);
  const candidatesBy = useMemo(() => new Map(result.candidates.map((r) => [r.s1Id, r.ids])), [result]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return businesses.filter((b) => (country === 'all' || b.country === country) && (!q || b.name.toLowerCase().includes(q) || b.id.toLowerCase().includes(q)));
  }, [businesses, country, query]);

  const selected = (selectedId && result.records.get(selectedId)) || filtered[0];
  const matched = new Set(selected ? (matchesBy.get(selected.id) ?? []) : []);
  const candidates = selected ? (candidatesBy.get(selected.id) ?? []) : [];
  const recordsFor = (source: 'S2' | 'S3') =>
    candidates
      .map((id) => result.records.get(id))
      .filter((r): r is BusinessRecord => !!r && r.source === source)
      .sort((a, b) => Number(matched.has(b.id)) - Number(matched.has(a.id)));

  return (
    <div className="grid gap-5 lg:grid-cols-[20rem_minmax(0,1fr)]">
      <div className="glass flex max-h-[36rem] min-w-0 flex-col rounded-card p-4">
        <div className="grid gap-2">
          <label className="text-xs text-muted" htmlFor="explorer-country">
            Country
          </label>
          <select
            id="explorer-country"
            value={country}
            onChange={(e) => setCountry(e.target.value)}
            className="h-10 rounded-control border border-line bg-canvas px-3 text-sm text-ink focus:border-glow focus:outline-none"
          >
            <option value="all">All countries</option>
            {countries.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search a business"
            aria-label="Search a business"
            className="h-10 rounded-control border border-line bg-canvas px-3 text-sm text-ink placeholder:text-muted focus:border-glow focus:outline-none"
          />
        </div>
        <p className="mt-3 text-xs text-muted">
          {filtered.length.toLocaleString('en-US')} businesses{filtered.length > LIST_LIMIT && `, showing ${LIST_LIMIT}; search to narrow`}
        </p>
        <ul className="mt-2 min-h-0 flex-1 overflow-y-auto" aria-label="Businesses">
          {filtered.slice(0, LIST_LIMIT).map((b) => (
            <li key={b.id}>
              <button
                type="button"
                aria-current={selected?.id === b.id}
                onClick={() => setSelectedId(b.id)}
                className={`w-full rounded-lg px-3 py-2 text-left text-sm ${selected?.id === b.id ? 'bg-accent/15 text-ink' : 'text-ink/85 hover:bg-raised/60'}`}
              >
                <span className="block truncate">{b.name}</span>
                <span className="text-xs text-muted">
                  {b.country} · {(matchesBy.get(b.id) ?? []).length} matches
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      {selected ? (
        <div className="min-w-0">
          <div className="rounded-card border border-accent/60 bg-surface p-5 shadow-glow">
            <div className="flex flex-wrap items-center gap-2">
              <Chip tone="accent">Source 1</Chip>
              <Chip>{selected.country}</Chip>
              <span className="font-mono text-xs text-muted">{selected.id}</span>
            </div>
            <p className="mt-3 text-h3 font-semibold break-words">{selected.name}</p>
            <p className="mt-1 text-sm break-words text-muted">{selected.address}</p>
            <p className="mt-3 text-xs text-muted">
              {matched.size} of {candidates.length} candidates matched by our pipeline. The test set has no labels, so these are predictions.
            </p>
          </div>
          {candidates.length === 0 ? (
            <p className="mt-5 text-sm text-muted">No candidate records were retrieved for this business.</p>
          ) : (
            <div className="mt-5 grid gap-5 md:grid-cols-2">
              {(['S2', 'S3'] as const).map((source) => {
                const list = recordsFor(source);
                return (
                  <section key={source} aria-label={`Source ${source.slice(1)} records`}>
                    <h3 className="text-sm font-medium text-muted">Source {source.slice(1)}</h3>
                    {list.length === 0 ? (
                      <p className="mt-2 text-sm text-muted">No candidates from this source.</p>
                    ) : (
                      <ul className="mt-2 grid gap-2">
                        {list.map((r) => (
                          <RecordCard key={r.id} record={r} matched={matched.has(r.id)} />
                        ))}
                      </ul>
                    )}
                  </section>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        <p className="text-muted">No business matches this filter.</p>
      )}
    </div>
  );
}
