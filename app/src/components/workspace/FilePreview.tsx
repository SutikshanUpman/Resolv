import type { ParsedTable } from '../../lib/parse/parseFile';

/** The first five rows of a parsed file. */
export function FilePreview({ table }: { table: ParsedTable }) {
  const rows = table.rows.slice(0, 5);
  return (
    <div className="overflow-x-auto rounded-lg border border-line" tabIndex={0} aria-label={`First ${rows.length} rows of ${table.fileName}`}>
      <table className="w-full text-left text-xs">
        <thead className="bg-raised/70 text-muted">
          <tr>
            {table.columns.map((c) => (
              <th key={c} className="px-2 py-1.5 font-medium whitespace-nowrap">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-line">
              {table.columns.map((c, j) => (
                <td key={c} className="max-w-48 truncate px-2 py-1.5 whitespace-nowrap text-ink/85" title={row[j]}>
                  {row[j]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
