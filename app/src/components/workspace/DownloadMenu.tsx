import { useEffect, useId, useRef, useState } from 'react';
import { challengeTsv, type ExportTable } from '../../lib/export/rows';
import { formatLabels, saveBlob, toCsv, toDocx, toPdf, toXlsx, type ExportFormat } from '../../lib/export/exporters';
import type { IdListRow } from '../../lib/resolver/types';
import { Button } from '../ui/Button';

interface DownloadMenuProps {
  table: ExportTable;
  /** Rows in the challenge's own format, for the .tsv option. */
  idRows: readonly IdListRow[];
}

const FORMATS: ExportFormat[] = ['csv', 'xlsx', 'pdf', 'docx', 'tsv'];

export function DownloadMenu({ table, idRows }: DownloadMenuProps) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<ExportFormat | null>(null);
  const [error, setError] = useState<string | null>(null);
  const menuId = useId();
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    root.current?.querySelector<HTMLButtonElement>('[role="menuitem"]')?.focus();
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !root.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('keydown', close);
    document.addEventListener('pointerdown', close);
    return () => {
      document.removeEventListener('keydown', close);
      document.removeEventListener('pointerdown', close);
    };
  }, [open]);

  const download = async (format: ExportFormat) => {
    setOpen(false);
    setBusy(format);
    setError(null);
    try {
      const blob =
        format === 'csv'
          ? toCsv(table)
          : format === 'xlsx'
            ? await toXlsx(table)
            : format === 'pdf'
              ? await toPdf(table)
              : format === 'docx'
                ? await toDocx(table)
                : new Blob([challengeTsv(idRows, table.kind)], { type: 'text/tab-separated-values' });
      const fileName = format === 'tsv' ? (table.kind === 'candidates' ? 'candidate_pairs.tsv' : 'matching_results.tsv') : `${table.name}.${format}`;
      saveBlob(blob, fileName);
    } catch (err) {
      setError(`The ${formatLabels[format]} export failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div ref={root} className="relative">
      <Button aria-haspopup="menu" aria-expanded={open} aria-controls={menuId} onClick={() => setOpen((o) => !o)} disabled={busy !== null}>
        {busy ? `Preparing ${formatLabels[busy]}…` : 'Download'}
        <svg viewBox="0 0 20 20" className="size-4" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M5 8l5 5 5-5" />
        </svg>
      </Button>
      {open && (
        <div id={menuId} role="menu" className="absolute left-0 z-20 mt-2 w-72 max-w-[calc(100vw-3rem)] rounded-xl border border-line bg-raised p-1.5 shadow-card">
          {FORMATS.map((f) => (
            <button
              key={f}
              type="button"
              role="menuitem"
              onClick={() => void download(f)}
              onKeyDown={(e) => {
                const items = [...(root.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? [])];
                const i = items.indexOf(e.currentTarget);
                if (e.key === 'ArrowDown') items[(i + 1) % items.length]?.focus();
                if (e.key === 'ArrowUp') items[(i - 1 + items.length) % items.length]?.focus();
              }}
              className="block w-full rounded-lg px-3 py-2 text-left text-sm text-ink/90 hover:bg-accent/15 focus:bg-accent/15 focus:outline-none"
            >
              {formatLabels[f]}
            </button>
          ))}
        </div>
      )}
      {error && (
        <p role="alert" className="mt-2 text-sm text-danger">
          {error}
        </p>
      )}
    </div>
  );
}
