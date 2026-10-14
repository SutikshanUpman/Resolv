import type { IdListRow, ResolveResult } from '../resolver/types';

export type FileKind = 'candidates' | 'matching';

export interface ExportTable {
  kind: FileKind;
  /** Base file name without extension. */
  name: string;
  title: string;
  columns: string[];
  rows: string[][];
}

const BASE = ['source1_entity_id', 'source1_name', 'source1_address', 'country', 'record_id', 'record_source', 'record_name', 'record_address'];

/** The readable format: one row per (business, record) pair, with names and addresses. */
export function pairTable(result: ResolveResult, kind: FileKind): ExportTable {
  const matched = new Set(result.matches.flatMap((r) => r.ids.map((id) => `${r.s1Id}|${id}`)));
  const source = kind === 'candidates' ? result.candidates : result.matches;
  const rows: string[][] = [];
  for (const { s1Id, ids } of source) {
    const s1 = result.records.get(s1Id);
    for (const id of ids) {
      const r = result.records.get(id);
      const row = [s1Id, s1?.name ?? '', s1?.address ?? '', s1?.country ?? '', id, r?.source ?? id.slice(0, 2), r?.name ?? '', r?.address ?? ''];
      if (kind === 'candidates') row.push(matched.has(`${s1Id}|${id}`) ? 'yes' : 'no');
      rows.push(row);
    }
  }
  return {
    kind,
    name: kind === 'candidates' ? 'candidates' : 'matching',
    title: kind === 'candidates' ? 'Candidate pairs' : 'Matches',
    columns: kind === 'candidates' ? [...BASE, 'matched'] : BASE,
    rows,
  };
}

/** The exact challenge format: one row per S1, IDs comma-separated, tab-separated file. */
export function challengeTsv(rows: readonly IdListRow[], kind: FileKind): string {
  const header = kind === 'candidates' ? 'source1_entity_id\tcandidate_entity_ids' : 'source1_entity_id\tmatched_entity_ids';
  return [header, ...rows.map((r) => `${r.s1Id}\t${r.ids.join(',')}`)].join('\n') + '\n';
}
