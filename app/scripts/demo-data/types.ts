import type { IndicScript } from '../lib/text.ts';

/** One row of a challenge source file, with the raw fields kept for byte-exact output. */
export interface SourceRow {
  id: string;
  name: string;
  /** Empty string when the record has no address. */
  address: string;
  country: string;
  fields: string[];
}

export type Bucket = '0' | '1-2' | '3-5' | '6-7' | '8+';
export const BUCKETS: readonly Bucket[] = ['0', '1-2', '3-5', '6-7', '8+'];

export function bucketOf(matches: number): Bucket {
  if (matches === 0) return '0';
  if (matches <= 2) return '1-2';
  if (matches <= 5) return '3-5';
  if (matches <= 7) return '6-7';
  return '8+';
}

/** A test S1 kept in a sampling pool, with its rows from both submitted files. */
export interface PoolItem {
  key: number;
  /** Row number in test_source1.tsv (0 = first data row), used to keep file order in the output. */
  line: number;
  s1: SourceRow;
  candidates: string[];
  matches: string[];
  bucket: Bucket;
}

export interface CountryStats {
  s1: number;
  emptyMatches: number;
  matchedPairs: number;
  buckets: Record<Bucket, number>;
}

export interface TestScan {
  s1Rows: number;
  candidatePairs: number;
  matchedPairs: number;
  s1WithoutCandidates: number;
  emptyMatchRows: number;
  maxCandidates: number;
  maxMatches: number;
  /** Rows where a matched ID is missing from the S1's candidate list. */
  matchesOutsideCandidates: number;
  /** Records assigned to more than one S1. */
  duplicateAssignments: number;
  matchHistogram: Record<string, number>;
  byCountry: Map<string, CountryStats>;
  pools: Map<string, PoolItem[]>;
}

export type VariantKind =
  | 'script'
  | 'domain'
  | 'legal'
  | 'honorific'
  | 'formatting'
  | 'order'
  | 'spelling'
  | 'shortened'
  | 'repeated'
  | 'noAddress'
  | 'addressFormat';

export interface ShowcaseRecord {
  id: string;
  source: 'S2' | 'S3';
  name: string;
  address: string | null;
  /** anyascii transliteration, present only for names in an Indic script. */
  latin: string | null;
  script: IndicScript | null;
  kinds: VariantKind[];
}

export interface ShowcaseExample {
  id: string;
  country: string;
  /** 'verified': train ground truth. 'pipeline': our test predictions (no labels exist). */
  provenance: 'verified' | 'pipeline';
  s1: { id: string; name: string; address: string };
  records: ShowcaseRecord[];
  kinds: VariantKind[];
  scripts: IndicScript[];
}
