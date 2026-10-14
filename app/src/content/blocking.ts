import { docText, fact } from './fact';

// §2.2 / §3.1: candidates per business at each step. Validation and test numbers are kept apart.
export const funnel = {
  union: fact(86.711, '86.711', '3.1', { context: '| union (input to the re-ranker) | 86.711 |', split: 'validation' }),
  afterCutValidation: fact(5.58, '5.58', '1', { context: 'leaves 5.58 candidates per S1 on validation', split: 'validation' }),
  afterCutTest: fact(6.1152, '6.1152', '3.1', { context: '10,594,837 pairs (6.1152 per S1)', split: 'test' }),
  matchesTest: fact(3.38, '~3.38', '2.2', { context: '~3.38 matches/S1 (test)', split: 'test', approx: true }),
} as const;

// §3.1 and §B (official validator): test-side totals of the two submitted files
export const candidateFile = {
  pairs: fact(10594837, '10,594,837', '3.1', { split: 'test' }),
  s1WithoutCandidates: fact(5354, '5,354', '3.1', { context: '5,354 S1 have no candidate', split: 'test' }),
  nonEmptyRows: fact(1727190, '1,727,190', 'B', { context: '5,354 empty and 1,727,190 non-empty rows', split: 'test' }),
} as const;

export const matchingFile = {
  matchedRecords: fact(5852231, '5,852,231', '4.4', { split: 'test' }),
  emptyRows: fact(100129, '100,129', 'B', { context: '100,129 empty and 1,632,415 non-empty rows', split: 'test' }),
  nonEmptyRows: fact(1632415, '1,632,415', 'B', { context: '100,129 empty and 1,632,415 non-empty rows', split: 'test' }),
} as const;

// §3.1 table: candidate sets on validation (100,000 S1)
export const candidateSets = [
  {
    name: docText('sparse keys, top-50', '3.1'),
    perS1: fact(49.771, '49.771', '3.1', { split: 'validation' }),
    pairRecall: fact(0.9575, '0.9575', '3.1', { split: 'validation' }),
    oracleF05: fact(0.98418, '0.98418', '3.1', { split: 'validation' }),
  },
  {
    name: docText('dense bi-encoder, top-40', '3.1'),
    perS1: fact(40.0, '40.0', '3.1', { context: '| dense bi-encoder, top-40 | 40.0 |', split: 'validation' }),
    pairRecall: fact(0.9977, '0.9977', '3.1', { context: '| 40.0 | 0.9977 |', split: 'validation' }),
    oracleF05: fact(0.99932, '0.99932', '3.1', { split: 'validation' }),
  },
  {
    name: docText('union (input to the re-ranker)', '3.1'),
    perS1: fact(86.711, '86.711', '3.1', { split: 'validation' }),
    pairRecall: fact(0.9986, '0.9986', '3.1', { context: '| 86.711 | 0.9986 |', split: 'validation' }),
    oracleF05: fact(0.99959, '0.99959', '3.1', { split: 'validation' }),
  },
  {
    name: docText('two-pass re-ranker cut (final)', '3.1'),
    perS1: fact(5.58, '5.58', '3.1', { context: '| two-pass re-ranker cut (final) | 5.58 |', split: 'validation' }),
    pairRecall: fact(0.9943, '0.9943', '3.1', { context: '| 5.58 | 0.9943 |', split: 'validation' }),
    oracleF05: fact(0.99846, '0.99846', '3.1', { context: '| 0.9943 | 0.99846 |', split: 'validation' }),
  },
] as const;

// §3.1 / §3.2 retrieval settings
export const retrieval = {
  sparseTopK: fact(50, '50', '3.1', { context: 'keep the 50 best S1→record pairs per S1' }),
  denseTopK: fact(40, '40', '3.1', { context: 'keeps the top 40 records per S1' }),
  denseReverseTopK: fact(3, '3', '3.1', { context: 'the top 3 S1 per record' }),
  keyFamilyCount: fact(11, '11', '2.2', { context: '11 families, IDF-weighted' }),
  postingCap: fact(300, '300', '3.2', { context: 'more than 300 postings' }),
  idfFormula: docText('idf = ln((n_r + 1) / (df + 1))', '3.2'),
} as const;

// §3.2 sparse key families
export const keyFamilies = [
  { key: 't', content: docText('name tokens ≥ 3 characters (first 6)', '3.2') },
  { key: 'n', content: docText('sorted name-token pairs (adjacent and skip-one)', '3.2') },
  { key: 'q', content: docText('whole squashed name (≥ 4 characters)', '3.2') },
  { key: 'p', content: docText('6-character prefix of the squashed name', '3.2') },
  { key: 'a', content: docText('alphabetic address tokens ≥ 4 characters (first 8)', '3.2') },
  { key: 'b', content: docText('address token bigrams (first 10)', '3.2') },
  { key: 'z', content: docText('ZIP / PIN code', '3.2') },
  { key: 'm', content: docText('address numbers with ≥ 3 digits, leading zeros removed (first 4)', '3.2') },
  { key: 'k', content: docText('phonetic consonant skeleton of the name', '3.2') },
  { key: 'c', content: docText('name token (or squashed name) × locality token', '3.2') },
  { key: 'h', content: docText('first address number × street or locality token', '3.2') },
] as const;

// §3.2 cut rule selected on rr_tune
export const cutRule = {
  kmax: fact(8, '8', '3.2', { context: 'kmax = 8' }),
  floor: fact(0.005, '0.005', '3.2', { context: 'floor = 0.005' }),
  pHigh: fact(0.15, '0.15', '3.2', { context: 'p_hi = 0.15' }),
  budget: fact(6.0, '6.0', '3.2', { context: 'budget of 6.0 candidates per S1' }),
} as const;
