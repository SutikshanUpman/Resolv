import type { IdListRow, ResolveSummary } from './types';

export function summarise(matches: readonly IdListRow[]): ResolveSummary {
  const businesses = matches.length;
  const total = matches.reduce((n, r) => n + r.ids.length, 0);
  const empty = matches.filter((r) => r.ids.length === 0).length;
  return {
    businesses,
    matches: total,
    avgMatches: businesses ? total / businesses : 0,
    unmatchedShare: businesses ? empty / businesses : 0,
  };
}

/** Parses candidate_pairs.tsv / matching_results.tsv text (header, then "S1-id<TAB>id,id,..."). */
export function parseIdListTsv(text: string): IdListRow[] {
  return text
    .split(/\r?\n/)
    .slice(1)
    .filter((l) => l.trim() !== '')
    .map((l) => {
      const [s1Id = '', list = ''] = l.split('\t');
      return { s1Id, ids: list ? list.split(',') : [] };
    });
}
