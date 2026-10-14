import { inputs } from '../lib/paths.ts';
import { BottomK, hashUnit } from '../lib/sampling.ts';
import { CANDIDATE_HEADER, MATCHING_HEADER, SOURCE_HEADER, TsvReader, detach, idList } from '../lib/tsv.ts';
import { BUCKETS, bucketOf, type Bucket, type CountryStats, type PoolItem, type SourceRow, type TestScan } from './types.ts';

/** A row to keep: its strings are detached from the file chunk (see detach). */
export function toSourceRow(raw: string[]): SourceRow {
  const fields = detach(raw);
  const [id = '', name = '', address = '', country = ''] = fields;
  return { id, name, address: address.trim(), country, fields };
}

/** Packs "S2-123" / "S3-123" into one number so millions of IDs fit in a typed array. */
function recordKey(id: string): number {
  const source = id.charCodeAt(1) - 48;
  if (!id.startsWith('S') || id[2] !== '-' || (source !== 2 && source !== 3)) throw new Error(`Not an S2/S3 id: ${id}`);
  return Number(id.slice(3)) * 4 + source;
}

/**
 * Pass 1: test_source1, candidate_pairs and matching_results in lock-step (their rows are in the same order).
 * Collects whole-file statistics for the doc checks and fills one sampling pool per country × match bucket.
 */
export async function scanTestSet(seed: number, poolSizes: Record<Bucket, number>): Promise<TestScan> {
  const s1File = await TsvReader.open(inputs.testS1, SOURCE_HEADER);
  const candFile = await TsvReader.open(inputs.candidates, CANDIDATE_HEADER);
  const matchFile = await TsvReader.open(inputs.matching, MATCHING_HEADER);

  const scan: TestScan = {
    s1Rows: 0,
    candidatePairs: 0,
    matchedPairs: 0,
    s1WithoutCandidates: 0,
    emptyMatchRows: 0,
    maxCandidates: 0,
    maxMatches: 0,
    matchesOutsideCandidates: 0,
    duplicateAssignments: 0,
    matchHistogram: {},
    byCountry: new Map(),
    pools: new Map(),
  };
  const pools = new Map<string, BottomK<PoolItem>>();
  // Every matched record, packed, to verify that no record is assigned to two S1.
  let assigned = new Float64Array(6_000_000);
  let assignedCount = 0;

  for (let line = 0; ; line++) {
    const [a, b, c] = await Promise.all([s1File.next(), candFile.next(), matchFile.next()]);
    if (a === null || b === null || c === null) {
      if (a !== null || b !== null || c !== null) throw new Error('The three files have different row counts');
      break;
    }
    const [id = '', , , country = ''] = a;
    if (b[0] !== id || c[0] !== id) {
      throw new Error(`Row ${line + 2}: S1 ids differ (${id}, ${b[0]}, ${c[0]}); the files are not aligned`);
    }
    const candidates = idList(b[1]);
    const matches = idList(c[1]);

    scan.s1Rows++;
    scan.candidatePairs += candidates.length;
    scan.matchedPairs += matches.length;
    if (candidates.length === 0) scan.s1WithoutCandidates++;
    if (matches.length === 0) scan.emptyMatchRows++;
    scan.maxCandidates = Math.max(scan.maxCandidates, candidates.length);
    scan.maxMatches = Math.max(scan.maxMatches, matches.length);
    scan.matchHistogram[matches.length] = (scan.matchHistogram[matches.length] ?? 0) + 1;

    if (matches.length > 0) {
      const candidateSet = new Set(candidates);
      if (matches.some((m) => !candidateSet.has(m))) scan.matchesOutsideCandidates++;
      for (const m of matches) {
        if (assignedCount === assigned.length) {
          const grown = new Float64Array(assigned.length * 2);
          grown.set(assigned);
          assigned = grown;
        }
        assigned[assignedCount++] = recordKey(m);
      }
    }

    const bucket = bucketOf(matches.length);
    let stats = scan.byCountry.get(country);
    if (!stats) {
      stats = { s1: 0, emptyMatches: 0, matchedPairs: 0, buckets: Object.fromEntries(BUCKETS.map((k) => [k, 0])) as CountryStats['buckets'] };
      scan.byCountry.set(country, stats);
    }
    stats.s1++;
    stats.matchedPairs += matches.length;
    if (matches.length === 0) stats.emptyMatches++;
    stats.buckets[bucket]++;

    const poolKey = `${country}|${bucket}`;
    let pool = pools.get(poolKey);
    if (!pool) {
      pool = new BottomK<PoolItem>(poolSizes[bucket]);
      pools.set(poolKey, pool);
    }
    const key = hashUnit(seed, id);
    if (pool.accepts(key)) {
      pool.offer({ key, line, s1: toSourceRow(a), candidates: detach(candidates), matches: detach(matches), bucket });
    }
  }

  const keys = assigned.subarray(0, assignedCount).sort();
  for (let i = 1; i < keys.length; i++) if (keys[i] === keys[i - 1]) scan.duplicateAssignments++;
  for (const [key, pool] of pools) scan.pools.set(key, pool.result());
  return scan;
}

/** Pass 2: which S1 each of the given records is matched to in the full submission. */
export async function findOwners(recordIds: ReadonlySet<string>): Promise<Map<string, string>> {
  const owners = new Map<string, string>();
  const file = await TsvReader.open(inputs.matching, MATCHING_HEADER);
  for await (const [s1 = '', list] of file.rows()) {
    for (const id of idList(list)) if (recordIds.has(id)) owners.set(id, detach(s1));
  }
  return owners;
}

/** Streams source files and keeps the rows whose IDs are wanted, in file order. Also counts all rows. */
export async function fetchRows(
  paths: readonly string[],
  wanted: ReadonlySet<string>,
): Promise<{ rows: Map<string, SourceRow>; rowCounts: number[] }> {
  const rows = new Map<string, SourceRow>();
  const rowCounts: number[] = [];
  for (const path of paths) {
    const file = await TsvReader.open(path, SOURCE_HEADER);
    let count = 0;
    for await (const fields of file.rows()) {
      count++;
      if (wanted.has(fields[0] ?? '')) {
        const row = toSourceRow(fields);
        rows.set(row.id, row);
      }
    }
    rowCounts.push(count);
  }
  return { rows, rowCounts };
}

export interface TrainCase {
  key: number;
  s1: SourceRow;
  matches: string[];
}

/** Train S1 pools per country, joined with their ground-truth matches. */
export async function sampleTrainTruth(seed: number, countries: readonly string[], perCountry: number): Promise<TrainCase[]> {
  const pools = new Map(countries.map((c) => [c, new BottomK<{ key: number; s1: SourceRow }>(perCountry)]));
  const s1File = await TsvReader.open(inputs.trainS1, SOURCE_HEADER);
  for await (const fields of s1File.rows()) {
    const pool = pools.get(fields[3] ?? '');
    const key = hashUnit(seed, fields[0] ?? '');
    if (pool?.accepts(key)) pool.offer({ key, s1: toSourceRow(fields) });
  }
  const picked = new Map<string, { key: number; s1: SourceRow }>();
  for (const pool of pools.values()) for (const item of pool.result()) picked.set(item.s1.id, item);

  const cases: TrainCase[] = [];
  const truth = await TsvReader.open(inputs.trainTruth, MATCHING_HEADER);
  for await (const [id = '', list] of truth.rows()) {
    const item = picked.get(id);
    if (item) cases.push({ ...item, matches: detach(idList(list)) });
  }
  return cases;
}
