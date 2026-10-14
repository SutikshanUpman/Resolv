import { summarise } from '../resolver/summary';
import type { BusinessRecord, IdListRow, ResolveResult, SourceId } from '../resolver/types';

const KEY = 'resolv.lastResult.v1';

type StoredRecord = [id: string, source: SourceId, name: string, address: string, country: string];

interface StoredResult {
  v: 1;
  mode: ResolveResult['mode'];
  candidates: [string, string][];
  matches: [string, string][];
  /**
   * null when the records did not fit in sessionStorage (about 2.7 million characters for the sample; Safari allows
   * less). Demo results always come from the sample files, so the records are then rebuilt from them.
   */
  records: StoredRecord[] | null;
}

const packRows = (rows: readonly IdListRow[]): [string, string][] => rows.map((r) => [r.s1Id, r.ids.join(',')]);
const unpackRows = (rows: [string, string][]): IdListRow[] => rows.map(([s1Id, ids]) => ({ s1Id, ids: ids ? ids.split(',') : [] }));

/** Saves the last result for this tab, so /app/results survives a reload. null clears it. */
export function saveResult(result: ResolveResult | null): void {
  try {
    if (!result) {
      sessionStorage.removeItem(KEY);
      return;
    }
    const lean: StoredResult = { v: 1, mode: result.mode, candidates: packRows(result.candidates), matches: packRows(result.matches), records: null };
    const records: StoredRecord[] = [...result.records.values()].map((r) => [r.id, r.source, r.name, r.address ?? '', r.country]);
    try {
      sessionStorage.setItem(KEY, JSON.stringify({ ...lean, records }));
    } catch {
      // Over quota: keep the IDs only, unless the records can't be rebuilt (a live result).
      if (result.mode === 'demo') sessionStorage.setItem(KEY, JSON.stringify(lean));
      else sessionStorage.removeItem(KEY);
    }
  } catch {
    // Storage blocked (private mode, disabled site data): the result simply won't survive a reload.
  }
}

export type StoredState =
  | { kind: 'none' }
  | { kind: 'complete'; result: ResolveResult }
  /** IDs restored; the records must be rebuilt from the sample files with completeResult(). */
  | { kind: 'needsRecords'; stored: StoredResult };

export function loadStoredResult(): StoredState {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return { kind: 'none' };
    const stored = JSON.parse(raw) as StoredResult;
    if (stored.v !== 1) return { kind: 'none' };
    if (!stored.records) return { kind: 'needsRecords', stored };
    const records = new Map<string, BusinessRecord>(
      stored.records.map(([id, source, name, address, country]) => [id, { id, source, name, address: address || null, country }]),
    );
    return { kind: 'complete', result: build(stored, records) };
  } catch {
    return { kind: 'none' };
  }
}

export function completeResult(stored: StoredResult, records: readonly BusinessRecord[]): ResolveResult {
  return build(stored, new Map(records.map((r) => [r.id, r])));
}

function build(stored: StoredResult, records: Map<string, BusinessRecord>): ResolveResult {
  const matches = unpackRows(stored.matches);
  return { mode: stored.mode, candidates: unpackRows(stored.candidates), matches, records, summary: summarise(matches) };
}
