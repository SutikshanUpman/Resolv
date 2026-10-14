import { useEffect, useRef } from 'react';
import type { ExportTable } from '../../lib/export/rows';
import { Button } from '../ui/Button';
import { VirtualTable } from './VirtualTable';

interface PreviewDialogProps {
  table: ExportTable | null;
  onClose: () => void;
}

/** Native modal dialog: focus is trapped and Escape closes it. */
export function PreviewDialog({ table, onClose }: PreviewDialogProps) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (table && !d.open) d.showModal();
    if (!table && d.open) d.close();
  }, [table]);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      aria-labelledby="preview-title"
      className="m-auto h-[min(90dvh,48rem)] w-[min(96vw,80rem)] max-w-none rounded-card border border-line bg-surface p-0 text-ink shadow-card backdrop:bg-ink/45"
    >
      {table && (
        <div className="flex h-full flex-col p-5 md:p-6">
          <div className="mb-4 flex items-center justify-between gap-4">
            <h2 id="preview-title" className="text-h3 font-semibold">
              {table.name}.csv <span className="text-sm font-normal text-muted">· {table.title}</span>
            </h2>
            <Button variant="ghost" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>
          <VirtualTable table={table} />
        </div>
      )}
    </dialog>
  );
}
