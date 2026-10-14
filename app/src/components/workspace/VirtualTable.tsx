import { useMemo, useRef, useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import type { ExportTable } from '../../lib/export/rows';

const ROW_HEIGHT = 36;

/** Searchable, sortable, virtualized view of an export table; renders only the visible rows. */
export function VirtualTable({ table }: { table: ExportTable }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState<{ col: number; dir: 1 | -1 } | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q ? table.rows.filter((r) => r.some((c) => c.toLowerCase().includes(q))) : table.rows;
    if (!sort) return filtered;
    const collator = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });
    return [...filtered].sort((a, b) => sort.dir * collator.compare(a[sort.col] ?? '', b[sort.col] ?? ''));
  }, [table.rows, query, sort]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scroller.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
  });

  const template = table.columns.map((c) => (c.includes('name') || c.includes('address') ? 'minmax(14rem,2fr)' : 'minmax(8.5rem,1fr)')).join(' ');

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search any column"
          aria-label="Search rows"
          className="h-10 w-full max-w-sm rounded-control border border-line bg-canvas px-3 text-sm text-ink placeholder:text-muted focus:border-glow focus:outline-none"
        />
        <p className="text-sm text-muted" aria-live="polite">
          {rows.length.toLocaleString('en-US')} of {table.rows.length.toLocaleString('en-US')} rows
        </p>
      </div>
      <div ref={scroller} className="mt-4 min-h-0 flex-1 overflow-auto rounded-lg border border-line" role="table" aria-rowcount={rows.length + 1}>
        <div style={{ gridTemplateColumns: template }} className="sticky top-0 z-10 grid min-w-max bg-raised text-xs" role="row">
          {table.columns.map((c, i) => {
            const active = sort?.col === i;
            return (
              <button
                key={c}
                type="button"
                role="columnheader"
                aria-sort={active ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}
                onClick={() => setSort(active && sort.dir === -1 ? null : { col: i, dir: active ? -1 : 1 })}
                className="flex items-center gap-1 border-b border-line px-3 py-2 text-left font-medium text-muted hover:text-ink"
              >
                {c}
                <span aria-hidden="true" className="text-glow">
                  {active ? (sort.dir === 1 ? '↑' : '↓') : ''}
                </span>
              </button>
            );
          })}
        </div>
        {rows.length === 0 ? (
          <p className="p-6 text-sm text-muted">No rows match “{query}”.</p>
        ) : (
          <div className="relative min-w-max" style={{ height: virtualizer.getTotalSize() }}>
            {virtualizer.getVirtualItems().map((v) => {
              const row = rows[v.index] ?? [];
              return (
                <div
                  key={v.key}
                  role="row"
                  style={{ gridTemplateColumns: template, transform: `translateY(${v.start}px)`, height: ROW_HEIGHT }}
                  className="absolute inset-x-0 top-0 grid border-b border-line text-xs odd:bg-surface/40"
                >
                  {row.map((cell, j) => (
                    <span key={j} role="cell" title={cell} className={`truncate px-3 py-2.5 ${j === 0 || table.columns[j]?.endsWith('_id') ? 'font-mono text-muted' : 'text-ink/90'}`}>
                      {cell}
                    </span>
                  ))}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
