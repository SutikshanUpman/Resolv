import {
  candidateFile,
  countryObservations,
  dataset,
  funnel,
  matchingFile,
  shift,
  type Fact,
} from '../../src/content/index.ts';
import type { TestScan } from './types.ts';

export interface Check {
  check: string;
  /** Where the expected value comes from. */
  source: string;
  expected: string;
  actual: string;
  pass: boolean;
}

function decimals(f: Fact): number {
  const n = f.quote.replace(/[,~≈%M\s]/g, '');
  const dot = n.indexOf('.');
  return dot < 0 ? 0 : n.length - dot - 1;
}

function exact(check: string, f: Fact, actual: number): Check {
  return { check, source: `§${f.section}: ${f.quote}`, expected: String(f.value), actual: String(actual), pass: actual === f.value };
}

/** Compares at the precision the doc states, e.g. 6.11519 against "6.1152". */
function rounded(check: string, f: Fact, actual: number): Check {
  const d = decimals(f);
  const expected = f.value.toFixed(d);
  return { check, source: `§${f.section}: ${f.quote}`, expected, actual: actual.toFixed(d), pass: actual.toFixed(d) === expected };
}

function none(check: string, source: string, actual: number): Check {
  return { check, source, expected: '0', actual: String(actual), pass: actual === 0 };
}

/** Our submitted files, re-measured, against the numbers the site shows. */
export function checkTestScan(scan: TestScan, recordRows: number): Check[] {
  const checks: Check[] = [
    exact('Test S1 rows', dataset.testS1, scan.s1Rows),
    exact('Test S2 + S3 records', dataset.testRecords, recordRows),
    exact('Candidate pairs', candidateFile.pairs, scan.candidatePairs),
    exact('S1 with no candidate', candidateFile.s1WithoutCandidates, scan.s1WithoutCandidates),
    exact('Non-empty candidate rows', candidateFile.nonEmptyRows, scan.s1Rows - scan.s1WithoutCandidates),
    rounded('Candidates per S1 (test)', funnel.afterCutTest, scan.candidatePairs / scan.s1Rows),
    exact('Matched records', matchingFile.matchedRecords, scan.matchedPairs),
    exact('Empty match rows', matchingFile.emptyRows, scan.emptyMatchRows),
    exact('Non-empty match rows', matchingFile.nonEmptyRows, scan.s1Rows - scan.emptyMatchRows),
    rounded('Matches per S1 (test)', funnel.matchesTest, scan.matchedPairs / scan.s1Rows),
    exact('France test S1', shift.franceTestS1, scan.byCountry.get('France')?.s1 ?? 0),
    none('Matches outside their candidate list', '§B: every match lies inside its S1’s candidate list', scan.matchesOutsideCandidates),
    none('Records assigned to two S1', '§4.4, §B: no record is assigned to two S1', scan.duplicateAssignments),
  ];
  for (const obs of countryObservations) {
    const stats = scan.byCountry.get(obs.country);
    const s1 = stats?.s1 ?? 0;
    checks.push(rounded(`${obs.country} matches per S1`, obs.matchesPerS1, s1 ? (stats?.matchedPairs ?? 0) / s1 : 0));
    checks.push(rounded(`${obs.country} share left empty (%)`, obs.emptyShare, s1 ? (100 * (stats?.emptyMatches ?? 0)) / s1 : 0));
  }
  return checks;
}
