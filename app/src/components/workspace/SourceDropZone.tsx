import { useId, useRef, useState, type DragEvent } from 'react';
import { clearSource, loadUpload } from '../../lib/workspace/loadSources';
import type { SlotState } from '../../lib/workspace/store';
import type { SourceId } from '../../lib/resolver/types';
import { Chip } from '../ui/Chip';
import { FilePreview } from './FilePreview';

interface SourceDropZoneProps {
  source: SourceId;
  label: string;
  role: string;
  slot: SlotState;
}

const kb = (bytes: number) => (bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`);

export function SourceDropZone({ source, label, role, slot }: SourceDropZoneProps) {
  const inputId = useId();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) void loadUpload(source, file);
  };

  const ok = slot.status === 'ready' && slot.validation.errors.length === 0;
  const bad = slot.status === 'error' || (slot.status === 'ready' && slot.validation.errors.length > 0);

  return (
    <section
      aria-label={`${label}: ${role}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`flex min-w-0 flex-col rounded-card border p-5 transition-colors ${
        dragging ? 'border-accent bg-accent/10' : ok ? 'border-match/40 bg-surface/60' : bad ? 'border-danger/40 bg-surface/60' : 'border-dashed border-line-strong bg-surface/40'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="font-semibold text-ink">{label}</h2>
          <p className="text-sm text-muted">{role}</p>
        </div>
        {ok && <Chip tone="match">Ready</Chip>}
        {bad && <Chip tone="danger">Needs attention</Chip>}
      </div>

      <input
        ref={input}
        id={inputId}
        type="file"
        accept=".csv,.tsv,.txt,.xlsx,.xls"
        className="sr-only"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void loadUpload(source, file);
          e.target.value = '';
        }}
      />

      {slot.status === 'empty' && (
        <label
          htmlFor={inputId}
          className="mt-5 grid flex-1 cursor-pointer place-items-center rounded-xl border border-line px-4 py-10 text-center text-sm text-muted transition-colors hover:border-line-strong hover:text-ink focus-within:border-glow"
        >
          <span>
            <span className="block font-medium text-ink">Drop a file or browse</span>
            CSV, TSV or Excel · columns entity_id, business_name, business_address, country
          </span>
        </label>
      )}

      {slot.status === 'parsing' && (
        <div className="mt-5 flex-1 rounded-xl border border-line p-4" aria-busy="true">
          <p className="text-sm text-muted">Reading {slot.fileName}…</p>
          <div className="mt-3 h-1.5 animate-pulse rounded-full bg-accent/40" />
        </div>
      )}

      {slot.status === 'error' && (
        <div className="mt-5 flex-1" role="alert">
          <p className="text-sm font-medium text-ink">{slot.fileName}</p>
          <p className="mt-2 text-sm text-danger">{slot.message}</p>
        </div>
      )}

      {slot.status === 'ready' && (
        <div className="mt-5 flex min-w-0 flex-1 flex-col gap-4">
          <div>
            <p className="truncate font-mono text-sm text-ink" title={slot.table.fileName}>
              {slot.table.fileName}
            </p>
            <p className="mt-1 text-xs text-muted">
              {slot.table.format} · {kb(slot.table.size)} · {slot.table.rows.length.toLocaleString('en-US')} rows
              {slot.origin === 'sample' && ' · sample data'}
            </p>
            <div className="mt-2 flex flex-wrap gap-1">
              {slot.table.columns.map((c) => (
                <Chip key={c}>{c}</Chip>
              ))}
            </div>
          </div>
          <FilePreview table={slot.table} />
          {slot.validation.errors.length > 0 && (
            <ul className="space-y-1 text-sm text-danger" role="alert">
              {slot.validation.errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          )}
          {slot.validation.warnings.map((w) => (
            <p key={w} className="text-sm text-unsure">
              {w}
            </p>
          ))}
          {slot.validation.errors.length === 0 && (
            <p className="text-sm text-match">Schema check passed: {slot.validation.records.length.toLocaleString('en-US')} records.</p>
          )}
        </div>
      )}

      {slot.status !== 'empty' && slot.status !== 'parsing' && (
        <div className="mt-4 flex gap-3 text-sm">
          <label htmlFor={inputId} className="cursor-pointer text-glow hover:underline">
            Replace file
          </label>
          <button type="button" onClick={() => clearSource(source)} className="text-muted hover:text-ink">
            Remove
          </button>
        </div>
      )}
    </section>
  );
}
