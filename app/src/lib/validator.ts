import type { ResolveResult } from './resolver/types';

export interface ValidatorCheck {
  rule: string;
  pass: boolean;
  detail: string;
}

/**
 * The official validator's rules (utils/validate_submission.py), run in the browser on both result files,
 * plus the exclusive-assignment rule from §4.4.
 */
export function validateResult(result: ResolveResult): ValidatorCheck[] {
  const s1Ids = [...result.records.values()].filter((r) => r.source === 'S1').map((r) => r.id);
  const recordIds = new Set([...result.records.values()].filter((r) => r.source !== 'S1').map((r) => r.id));
  const checks: ValidatorCheck[] = [];

  for (const [name, rows] of [
    ['candidates', result.candidates],
    ['matching', result.matches],
  ] as const) {
    const seen = new Set<string>();
    let dupRows = 0;
    let dupIds = 0;
    let badPrefix = 0;
    let unknown = 0;
    for (const r of rows) {
      if (seen.has(r.s1Id)) dupRows++;
      seen.add(r.s1Id);
      if (new Set(r.ids).size !== r.ids.length) dupIds++;
      for (const id of r.ids) {
        if (!id.startsWith('S2-') && !id.startsWith('S3-')) badPrefix++;
        else if (!recordIds.has(id)) unknown++;
      }
    }
    const missing = s1Ids.filter((id) => !seen.has(id)).length;
    checks.push(
      { rule: `${name}: one row per Source 1 business`, pass: missing === 0 && dupRows === 0, detail: `${rows.length.toLocaleString('en-US')} rows, ${missing} missing, ${dupRows} duplicated` },
      { rule: `${name}: no repeated ID within a list`, pass: dupIds === 0, detail: `${dupIds} lists with repeats` },
      { rule: `${name}: only S2/S3 IDs that exist in the sources`, pass: badPrefix === 0 && unknown === 0, detail: `${badPrefix} wrong prefix, ${unknown} unknown` },
    );
  }

  const candidates = new Map(result.candidates.map((r) => [r.s1Id, new Set(r.ids)]));
  const outside = result.matches.reduce((n, r) => n + r.ids.filter((id) => !candidates.get(r.s1Id)?.has(id)).length, 0);
  checks.push({ rule: 'Every match is one of its business’s candidates', pass: outside === 0, detail: `${outside} matches outside the candidate list` });

  const owner = new Map<string, number>();
  for (const r of result.matches) for (const id of r.ids) owner.set(id, (owner.get(id) ?? 0) + 1);
  const shared = [...owner.values()].filter((n) => n > 1).length;
  checks.push({ rule: 'Each record is assigned to at most one business', pass: shared === 0, detail: `${shared} records assigned twice` });
  return checks;
}
