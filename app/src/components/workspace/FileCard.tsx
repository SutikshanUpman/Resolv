import type { ExportTable } from '../../lib/export/rows';
import type { IdListRow } from '../../lib/resolver/types';
import { Button } from '../ui/Button';
import { DownloadMenu } from './DownloadMenu';

interface FileCardProps {
  table: ExportTable;
  idRows: readonly IdListRow[];
  description: string;
  onPreview: () => void;
}

export function FileCard({ table, idRows, description, onPreview }: FileCardProps) {
  return (
    <article className="glass flex flex-col rounded-card p-6">
      <div className="flex items-center gap-3">
        <span aria-hidden="true" className="grid size-10 place-items-center rounded-lg border border-accent/40 bg-accent/10 font-mono text-xs text-glow">
          CSV
        </span>
        <div className="min-w-0">
          <h3 className="truncate font-mono text-ink">{table.name}.csv</h3>
          <p className="text-sm text-muted">{table.rows.length.toLocaleString('en-US')} rows · one per business and record</p>
        </div>
      </div>
      <p className="mt-4 flex-1 text-sm text-muted">{description}</p>
      <div className="mt-6 flex flex-wrap gap-3">
        <Button variant="secondary" onClick={onPreview}>
          Preview
        </Button>
        <DownloadMenu table={table} idRows={idRows} />
      </div>
    </article>
  );
}
