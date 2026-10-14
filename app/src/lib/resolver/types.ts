import type { StageId } from '../../content/pipeline';

export type SourceId = 'S1' | 'S2' | 'S3';

export interface BusinessRecord {
  id: string;
  source: SourceId;
  name: string;
  /** null when the record has no address. */
  address: string | null;
  country: string;
}

export interface ResolverInput {
  /** 'sample': our precomputed demo data. 'upload': the visitor's own files. */
  origin: 'sample' | 'upload';
  s1: BusinessRecord[];
  s2: BusinessRecord[];
  s3: BusinessRecord[];
}

/** One row of candidate_pairs.tsv or matching_results.tsv: an S1 and its list of S2/S3 IDs. */
export interface IdListRow {
  s1Id: string;
  ids: string[];
}

/** Counters with real values for the run. Stages without one show the doc's labelled rate instead. */
export type CounterKey = 'businesses' | 'records' | 'candidatesAfterCut' | 'matchesAssigned' | 'unmatched';

export type ResolverErrorCode = 'LIVE_INFERENCE_UNAVAILABLE' | 'NOT_CONFIGURED' | 'LOAD_FAILED' | 'INPUT_MISMATCH';

export interface ResolveSummary {
  businesses: number;
  matches: number;
  avgMatches: number;
  /** Share of businesses with no match, 0 to 1. */
  unmatchedShare: number;
}

export interface ResolveResult {
  mode: 'demo' | 'live';
  candidates: IdListRow[];
  matches: IdListRow[];
  records: ReadonlyMap<string, BusinessRecord>;
  summary: ResolveSummary;
}

export type ResolverEvent =
  | { type: 'stage-start'; stage: StageId }
  | { type: 'progress'; stage: StageId; counter: CounterKey; value: number; total: number }
  | { type: 'stage-end'; stage: StageId }
  | { type: 'done'; result: ResolveResult }
  | { type: 'error'; code: ResolverErrorCode; message: string };

export interface ResolverRun {
  events: AsyncIterable<ResolverEvent>;
  /** Fast-forward to the end; the remaining events arrive at once. */
  skip(): void;
  cancel(): void;
}

export interface Resolver {
  readonly kind: 'demo' | 'api';
  /** Whether the resolver can run models on a visitor's own files. */
  readonly liveInference: boolean;
  resolve(input: ResolverInput): ResolverRun;
}
