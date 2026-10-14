import type { BusinessRecord, SourceId } from '../resolver/types';
import type { ParsedTable } from './parseFile';

/** The challenge's input schema (README, "Data Description"), the same for all three sources. */
export const REQUIRED_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country'] as const;

export interface Validation {
  errors: string[];
  warnings: string[];
  records: BusinessRecord[];
}

const examples = (ids: string[]) => ids.slice(0, 3).join(', ') + (ids.length > 3 ? ` and ${ids.length - 3} more` : '');

/** Checks a parsed file against the schema for its source and turns valid rows into records. */
export function validateSource(table: ParsedTable, source: SourceId): Validation {
  const errors: string[] = [];
  const warnings: string[] = [];
  const index = new Map(table.columns.map((c, i) => [c.toLowerCase().replace(/\s+/g, '_'), i]));
  const missing = REQUIRED_COLUMNS.filter((c) => !index.has(c));
  if (missing.length > 0) {
    errors.push(
      `Missing column${missing.length > 1 ? 's' : ''}: ${missing.join(', ')}. Each source needs ${REQUIRED_COLUMNS.join(', ')}; found ${table.columns.join(', ') || 'none'}.`,
    );
    return { errors, warnings, records: [] };
  }
  if (table.rows.length === 0) {
    errors.push('The file has a header but no rows.');
    return { errors, warnings, records: [] };
  }

  const col = (row: string[], name: (typeof REQUIRED_COLUMNS)[number]) => (row[index.get(name) ?? -1] ?? '').trim();
  const seen = new Set<string>();
  const wrongPrefix: string[] = [];
  const duplicates: string[] = [];
  const noId: number[] = [];
  const noName: string[] = [];
  const noCountry: string[] = [];
  let noAddress = 0;
  const records: BusinessRecord[] = [];

  table.rows.forEach((row, i) => {
    const id = col(row, 'entity_id');
    if (!id) {
      noId.push(i + 2);
      return;
    }
    if (!id.startsWith(`${source}-`)) wrongPrefix.push(id);
    if (seen.has(id)) duplicates.push(id);
    seen.add(id);
    const name = col(row, 'business_name');
    const address = col(row, 'business_address');
    const country = col(row, 'country');
    if (!name) noName.push(id);
    if (!country) noCountry.push(id);
    if (!address) noAddress++;
    records.push({ id, source, name, address: address || null, country });
  });

  if (noId.length) errors.push(`${noId.length} row(s) have no entity_id (e.g. line ${noId.slice(0, 3).join(', ')}).`);
  if (wrongPrefix.length) errors.push(`${wrongPrefix.length} ID(s) don’t start with “${source}-”, so this may be the wrong source: ${examples(wrongPrefix)}.`);
  if (duplicates.length) errors.push(`${duplicates.length} duplicate ID(s): ${examples(duplicates)}.`);
  if (noName.length) errors.push(`${noName.length} record(s) have no business_name: ${examples(noName)}.`);
  if (noCountry.length) errors.push(`${noCountry.length} record(s) have no country: ${examples(noCountry)}.`);
  if (noAddress) {
    (source === 'S1' ? errors : warnings).push(
      source === 'S1'
        ? `${noAddress.toLocaleString('en-US')} reference record(s) have no business_address; Source 1 addresses must be present.`
        : `${noAddress.toLocaleString('en-US')} record(s) have no address. That’s allowed; they are matched on name alone.`,
    );
  }
  return { errors, warnings, records };
}
