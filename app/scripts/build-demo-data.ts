/**
 * Builds public/demo/ from the real submission. Every file is streamed; nothing large is held in memory.
 *
 *   npm run demo-data
 *
 * Inputs (read-only): our two submitted TSVs, which hold IDs only, and the challenge's test and train
 * source files, which hold the names and addresses behind those IDs. The script first re-measures the
 * submitted files and stops if any number differs from the doc. Outputs:
 *
 *   sample_source{1,2,3}.tsv   3,000 test S1 (1,000 per country) and all their candidate records, in the challenge's input format
 *   candidate_pairs.tsv        our submitted candidate rows for those S1
 *   matching_results.tsv       our submitted match rows for those S1
 *   trace.json                 for candidates not matched to their S1: the S1 they were assigned to instead
 *   showcase.json              landing-page examples: train ground truth (India, US) and test predictions (France)
 *   manifest.json              seed, sampling design, whole-file statistics and every check
 */
import { mkdirSync, statSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { DEMO_DIR, inputs } from './lib/paths.ts';
import { nameParts, skeleton } from './lib/text.ts';
import { CANDIDATE_HEADER, MATCHING_HEADER, SOURCE_HEADER } from './lib/tsv.ts';
import { checkTestScan } from './demo-data/checks.ts';
import { bucketTargets, selectCountry, type CountrySelection, type QuotaRule } from './demo-data/select.ts';
import { buildExample, pickExamples } from './demo-data/showcase.ts';
import { fetchRows, findOwners, sampleTrainTruth, scanTestSet } from './demo-data/streams.ts';
import type { Bucket, ShowcaseExample, SourceRow } from './demo-data/types.ts';

const SEED = 42;
const PER_COUNTRY = 1000;
/** Over-represented on purpose so the demo shows abstentions and heavy businesses. */
const FIXED_TARGETS: Partial<Record<Bucket, number>> = { '0': 100, '8+': 60 };
/** Pool sizes per country and bucket: comfortably above any target, to leave room for quota swaps. */
const POOL_SIZES: Record<Bucket, number> = { '0': 400, '1-2': 1000, '3-5': 2000, '6-7': 500, '8+': 300 };
const QUOTAS: Readonly<Record<string, QuotaRule>> = {
  India: { flag: 'indicMatch', min: 300 },
  France: { flag: 'acronymCandidate', min: 80 },
};
const SHOWCASE = { India: 4, US: 3, France: 3 } as const;
const TRAIN_POOL_PER_COUNTRY = 2000;

const started = Date.now();
const log = (msg: string) => console.log(`[${((Date.now() - started) / 1000).toFixed(1).padStart(6)} s] ${msg}`);

function tsv(header: readonly string[], rows: readonly (readonly string[])[]): string {
  return [header, ...rows].map((r) => r.join('\t')).join('\n') + '\n';
}

function write(name: string, body: string): number {
  const path = join(DEMO_DIR, name);
  writeFileSync(path, body, 'utf8');
  return statSync(path).size;
}

/** The doc's own normalisation example (§4.2.1), to prove the port of the name rules before using it. */
function selfTestNameRules(): void {
  const cases = [
    { name: 'Anand Foods Private Limited', core: 'anand foods' },
    { name: 'আনন্দ ফুডস প্রাইভেট লিমিটেড', core: 'annd phuds' },
  ];
  for (const c of cases) {
    const core = nameParts(c.name).core.join(' ');
    const sk = skeleton(core);
    if (core !== c.core || sk !== 'and fds') {
      throw new Error(`Name-rule port disagrees with §4.2.1: "${c.name}" gave core "${core}", skeleton "${sk}"`);
    }
  }
}

async function main(): Promise<void> {
  selfTestNameRules();
  mkdirSync(DEMO_DIR, { recursive: true });

  // Pass 1: S1 + both submitted files in lock-step.
  log('Pass 1: scanning test S1 with candidate_pairs.tsv and matching_results.tsv');
  const scan = await scanTestSet(SEED, POOL_SIZES);
  const pooled = [...scan.pools.values()].flat();
  const pooledRecordIds = new Set(pooled.flatMap((p) => p.candidates));
  log(`  ${scan.s1Rows.toLocaleString('en-US')} S1, ${pooled.length.toLocaleString('en-US')} pooled, ${pooledRecordIds.size.toLocaleString('en-US')} candidate records to fetch`);

  // Pass 2: names and addresses of the pooled candidates; also counts every S2/S3 row.
  log('Pass 2: fetching candidate records from test S2 and S3');
  const { rows: records, rowCounts } = await fetchRows([inputs.testS2, inputs.testS3], pooledRecordIds);
  const recordRows = rowCounts.reduce((a, b) => a + b, 0);
  const missing = [...pooledRecordIds].filter((id) => !records.has(id));
  if (missing.length > 0) throw new Error(`${missing.length} candidate IDs not found in S2/S3, e.g. ${missing.slice(0, 3).join(', ')}`);

  const checks = checkTestScan(scan, recordRows);
  for (const c of checks) log(`  ${c.pass ? 'ok  ' : 'FAIL'} ${c.check}: ${c.actual} (doc ${c.expected})`);
  const failed = checks.filter((c) => !c.pass);
  if (failed.length > 0) throw new Error(`${failed.length} check(s) against the doc failed; nothing was written`);

  // Selection: stratified per country, then quotas for cross-script India and French acronyms.
  const selections: CountrySelection[] = [];
  for (const [country, stats] of [...scan.byCountry].sort(([a], [b]) => a.localeCompare(b))) {
    const targets = bucketTargets(stats, PER_COUNTRY, FIXED_TARGETS);
    selections.push(selectCountry(country, scan.pools, targets, records, QUOTAS[country]));
  }
  const selected = selections.flatMap((s) => s.selected).sort((a, b) => a.line - b.line);
  const selectedIds = new Set(selected.map((s) => s.s1.id));
  const sampleRecordIds = new Set(selected.flatMap((s) => s.candidates));

  // Pass 3: for candidates not matched to their own S1, which S1 received them instead.
  log('Pass 3: finding where unmatched candidates were assigned');
  const notMatchedHere = new Set(selected.flatMap((s) => s.candidates.filter((c) => !s.matches.includes(c))));
  const owners = await findOwners(notMatchedHere);
  const outsideOwnerIds = new Set([...owners.values()].filter((id) => !selectedIds.has(id)));
  log(`  ${owners.size.toLocaleString('en-US')} of ${notMatchedHere.size.toLocaleString('en-US')} went to another S1; fetching ${outsideOwnerIds.size.toLocaleString('en-US')} of those S1`);
  const { rows: outsideS1 } = await fetchRows([inputs.testS1], outsideOwnerIds);

  // Pass 4: train ground truth for the verified landing-page examples (India and US).
  log('Pass 4: sampling train ground truth for verified examples');
  const trainCases = await sampleTrainTruth(SEED, ['India', 'US'], TRAIN_POOL_PER_COUNTRY);
  const trainRecordIds = new Set(trainCases.flatMap((c) => c.matches));
  const { rows: trainRecords } = await fetchRows([inputs.trainS2, inputs.trainS3], trainRecordIds);

  const candidatesFor = (country: string): ShowcaseExample[] => {
    if (country === 'France') {
      return selected
        .filter((s) => s.s1.country === 'France')
        .sort((a, b) => a.key - b.key)
        .map((s) => buildExample(s.s1, recordsOf(s.matches, records), { provenance: 'pipeline', sameHouseNumber: true, needsScript: false }))
        .filter((e): e is ShowcaseExample => e !== null);
    }
    return trainCases
      .filter((c) => c.s1.country === country)
      .sort((a, b) => a.key - b.key)
      .map((c) => buildExample(c.s1, recordsOf(c.matches, trainRecords), { provenance: 'verified', sameHouseNumber: false, needsScript: country === 'India' }))
      .filter((e): e is ShowcaseExample => e !== null);
  };
  const picks = Object.fromEntries(
    Object.entries(SHOWCASE).map(([country, n]) => {
      const pool = candidatesFor(country);
      if (pool.length < n) throw new Error(`Only ${pool.length} clear ${country} examples, need ${n}`);
      log(`  ${country}: ${pool.length} clear examples, picking ${n}`);
      return [country, pickExamples(pool, n)];
    }),
  );
  const showcase: ShowcaseExample[] = [];
  for (let i = 0; showcase.length < Object.values(SHOWCASE).reduce((a, b) => a + b, 0); i++) {
    for (const country of ['India', 'France', 'US']) {
      const ex = picks[country]?.[i];
      if (ex) showcase.push(ex);
    }
  }

  // Write.
  log('Writing public/demo/');
  const sources = { S2: [] as string[][], S3: [] as string[][] };
  for (const [id, row] of records) if (sampleRecordIds.has(id)) (id.startsWith('S2-') ? sources.S2 : sources.S3).push(row.fields);
  const sizes: Record<string, number> = {
    'sample_source1.tsv': write('sample_source1.tsv', tsv(SOURCE_HEADER, selected.map((s) => s.s1.fields))),
    'sample_source2.tsv': write('sample_source2.tsv', tsv(SOURCE_HEADER, sources.S2)),
    'sample_source3.tsv': write('sample_source3.tsv', tsv(SOURCE_HEADER, sources.S3)),
    'candidate_pairs.tsv': write('candidate_pairs.tsv', tsv(CANDIDATE_HEADER, selected.map((s) => [s.s1.id, s.candidates.join(',')]))),
    'matching_results.tsv': write('matching_results.tsv', tsv(MATCHING_HEADER, selected.map((s) => [s.s1.id, s.matches.join(',')]))),
  };

  const trace = {
    description:
      'For each candidate of a sampled S1 that was not matched to it: the S1 our submission assigned it to instead. Candidates absent here were matched to no S1.',
    assignedElsewhere: Object.fromEntries([...owners].sort(([a], [b]) => a.localeCompare(b))),
    otherS1: Object.fromEntries(
      [...outsideS1.values()].map((r) => [r.id, { name: r.name, address: r.address, country: r.country }]),
    ),
  };
  sizes['trace.json'] = write('trace.json', JSON.stringify(trace));
  sizes['showcase.json'] = write(
    'showcase.json',
    JSON.stringify(
      {
        description:
          'Landing-page examples. "verified": train ground truth. "pipeline": matched by our pipeline on the test set, which has no labels. Chosen by rule in scripts/demo-data/showcase.ts.',
        examples: showcase,
      },
      null,
      1,
    ),
  );

  const manifest = {
    description: 'Demo data for the showcase site, sampled from the real submission by scripts/build-demo-data.ts.',
    seed: SEED,
    inputs: Object.fromEntries(
      Object.entries(inputs).map(([k, p]) => [k, { file: basename(p), bytes: statSync(p).size }]),
    ),
    fullTestSet: {
      s1: scan.s1Rows,
      records: recordRows,
      s2Records: rowCounts[0],
      s3Records: rowCounts[1],
      candidatePairs: scan.candidatePairs,
      matchedPairs: scan.matchedPairs,
      s1WithoutCandidates: scan.s1WithoutCandidates,
      emptyMatchRows: scan.emptyMatchRows,
      candidatesPerS1: round(scan.candidatePairs / scan.s1Rows, 5),
      matchesPerS1: round(scan.matchedPairs / scan.s1Rows, 5),
      maxCandidates: scan.maxCandidates,
      maxMatches: scan.maxMatches,
      matchHistogram: scan.matchHistogram,
      byCountry: Object.fromEntries(
        [...scan.byCountry].map(([c, s]) => [
          c,
          { s1: s.s1, matchesPerS1: round(s.matchedPairs / s.s1, 5), emptyShare: round(s.emptyMatches / s.s1, 5), buckets: s.buckets },
        ]),
      ),
    },
    checks,
    sample: {
      design:
        'Per country, 1,000 S1 drawn at random (seeded hash) within match-count buckets. Singletons (100) and businesses with 8+ matches (60) are over-represented; the other buckets follow the country’s real distribution. India is topped up to at least 300 businesses with a match in an Indic script, France to at least 80 with an acronym-like candidate, by swapping within the largest buckets. Averages over the sample therefore differ from the full test set.',
      s1: selected.length,
      s2Records: sources.S2.length,
      s3Records: sources.S3.length,
      candidatePairs: selected.reduce((n, s) => n + s.candidates.length, 0),
      matchedPairs: selected.reduce((n, s) => n + s.matches.length, 0),
      byCountry: Object.fromEntries(
        selections.map((s) => [s.country, { s1: s.selected.length, targets: s.targets, quota: s.quota ?? null }]),
      ),
    },
    showcase: {
      examples: showcase.length,
      verified: showcase.filter((e) => e.provenance === 'verified').length,
      pipeline: showcase.filter((e) => e.provenance === 'pipeline').length,
    },
    files: sizes,
  };
  sizes['manifest.json'] = write('manifest.json', JSON.stringify(manifest, null, 1));

  for (const [file, bytes] of Object.entries(sizes)) log(`  ${file.padEnd(22)} ${(bytes / 1024).toFixed(1).padStart(8)} KB`);
  log('Done');
}

function recordsOf(ids: readonly string[], rows: ReadonlyMap<string, SourceRow>): SourceRow[] {
  return ids.map((id) => rows.get(id)).filter((r): r is SourceRow => r !== undefined);
}

function round(x: number, digits: number): number {
  return Number(x.toFixed(digits));
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
