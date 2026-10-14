import { docText, fact, type DocSection, type DocText, type Fact } from './fact';
import { funnel, matchingFile } from './blocking';
import { headline } from './metrics';
import type { ModelId } from './models';

export type StageId =
  | 'normalise'
  | 'retrieve'
  | 'union'
  | 'rerankCut'
  | 'pairFeatures'
  | 'crossEncoder'
  | 'stacker'
  | 'llmRerank'
  | 'collective'
  | 'assign';

export interface StageMetric {
  readonly label: string;
  readonly fact: Fact;
}

export interface PipelineStage {
  readonly id: StageId;
  /** Position in the §2.2 diagram (stage 2 covers [2a] sparse and [2b] dense). */
  readonly order: number;
  readonly title: string;
  /** The stage's line in the §2.2 diagram. */
  readonly diagram: DocText;
  readonly summary: string;
  readonly models: readonly ModelId[];
  readonly metrics: readonly StageMetric[];
  /** Where the doc explains the stage in depth. */
  readonly sections: readonly DocSection[];
  /** File written at the end of this stage, if any. */
  readonly output?: 'candidate_pairs.tsv' | 'matching_results.tsv';
}

// §2.2 solution strategy: the ten stages, with numbers from §3 (blocking), §4.1 (architecture) and §4.4 (decision).
export const pipeline: readonly PipelineStage[] = [
  {
    id: 'normalise',
    order: 1,
    title: 'Normalisation',
    diagram: docText('[1] Normalisation (rules; anyascii transliteration)', '2.2'),
    summary:
      'Rules clean every name and address: Indic scripts are transliterated with anyascii, legal forms and honorifics are canonicalised, and a phonetic consonant skeleton lets native-script and Latin spellings meet.',
    models: [],
    metrics: [
      { label: 'India S2 names in Indic script', fact: fact(23.64, '23.64 %', '2.1', { context: '23.64 % (test)', split: 'test' }) },
    ],
    sections: ['4.2.1'],
  },
  {
    id: 'retrieve',
    order: 2,
    title: 'Two retrievers',
    diagram: docText('Sparse keys: 11 families, IDF-weighted, top-50 per S1', '2.2'),
    summary:
      'Within each country, IDF-weighted sparse keys keep the 50 best records per business, and a fine-tuned multilingual bi-encoder reading the raw scripts keeps the top 40 records per business and the top 3 businesses per record.',
    models: ['e5Small'],
    metrics: [
      { label: 'Pair recall, sparse top-50', fact: fact(0.9575, '0.9575', '3.1', { split: 'validation' }) },
      { label: 'Pair recall, dense top-40', fact: fact(0.9977, '0.9977', '3.1', { context: '| 40.0 | 0.9977 |', split: 'validation' }) },
    ],
    sections: ['3.1', '3.2'],
  },
  {
    id: 'union',
    order: 3,
    title: 'Union',
    diagram: docText('[3] Union', '2.2'),
    summary: 'The two candidate lists are merged. Together they contain almost every true pair.',
    models: [],
    metrics: [
      { label: 'Candidates per business', fact: funnel.union },
      { label: 'Pair recall', fact: fact(0.9986, '0.9986', '3.1', { context: '| 86.711 | 0.9986 |', split: 'validation' }) },
      { label: 'Oracle F0.5 (ceiling for the matcher)', fact: fact(0.99959, '0.99959', '3.1', { split: 'validation' }) },
    ],
    sections: ['3.1'],
  },
  {
    id: 'rerankCut',
    order: 4,
    title: 'Two-pass re-ranker and cut',
    diagram: docText('[4] Two-pass LightGBM re-ranker + cut', '2.2'),
    summary:
      'A LightGBM on 39 cheap features scores every candidate. A second pass adds 13 hints, including how this business ranks among all businesses claiming the same record, then a cut keeps a short list.',
    models: ['lightgbm'],
    metrics: [
      { label: 'Candidates per business', fact: funnel.afterCutValidation },
      { label: 'Candidates per business', fact: funnel.afterCutTest },
      { label: 'Pair recall', fact: fact(0.9943, '0.9943', '3.1', { context: '| 5.58 | 0.9943 |', split: 'validation' }) },
      { label: 'Oracle F0.5', fact: fact(0.99846, '0.99846', '3.1', { context: '| 0.9943 | 0.99846 |', split: 'validation' }) },
    ],
    sections: ['3.2'],
    output: 'candidate_pairs.tsv',
  },
  {
    id: 'pairFeatures',
    order: 5,
    title: 'Pair features and level-1 model',
    diagram: docText('[5] 107 pair features -> level-1 LightGBM -> p1 (+ 10 competition features)', '2.2'),
    summary:
      'Each candidate pair gets 107 features: string and address similarities, blocking evidence and how common the name is. A level-1 LightGBM turns them into a first match probability.',
    models: ['lightgbm'],
    metrics: [
      { label: 'Pair features', fact: fact(107, '107', '4.2.2', { context: '| Pair features (level-1 input) | 107 |' }) },
      { label: 'Macro F0.5, standalone', fact: fact(0.98466, '0.98466', '4.1', { context: '| 0.98466 (standalone) |', split: 'validation' }) },
    ],
    sections: ['4.2.2', '4.2.3'],
  },
  {
    id: 'crossEncoder',
    order: 6,
    title: 'Cross-encoder',
    diagram: docText('[6] Cross-encoder multilingual-e5-base (278M) -> logit (+ 9 derived features)', '2.2'),
    summary:
      'A multilingual cross-encoder reads both records side by side in their raw scripts and scores how likely they are the same business.',
    models: ['e5Base'],
    metrics: [
      { label: 'Training pairs', fact: fact(2137708, '2,137,708', '4.1') },
      { label: 'Macro F0.5, standalone', fact: fact(0.98545, '0.98545', '4.1', { context: '| 0.98545 (standalone) |', split: 'validation' }) },
    ],
    sections: ['4.1', '4.2.6'],
  },
  {
    id: 'stacker',
    order: 7,
    title: 'Stacker',
    diagram: docText('[7] LightGBM stacker (126 features) -> p', '2.2'),
    summary:
      'A LightGBM stacker combines the pair features, the level-1 competition features and the cross-encoder scores into one probability.',
    models: ['lightgbm'],
    metrics: [
      { label: 'Stacker features', fact: fact(126, '126', '4.2.2', { context: '| Stacker input | 126 |' }) },
      { label: 'Macro F0.5, cumulative', fact: fact(0.99093, '0.99093', '4.1', { context: '| 0.99093 (cumulative) |', split: 'validation' }) },
    ],
    sections: ['4.1', '4.2.4'],
  },
  {
    id: 'llmRerank',
    order: 8,
    title: 'LLM reranker on uncertain pairs',
    diagram: docText('[8] Qwen3-Reranker-0.6B (596M) on uncertain pairs (v3a stacker p in [0.02, 0.98])', '2.2'),
    summary:
      'A small multilingual LLM reranker, fine-tuned on our training pairs, scores only the pairs the earlier models are unsure about. It brings knowledge of acronyms, French abbreviations and trade names.',
    models: ['qwen'],
    metrics: [
      { label: 'Training pairs', fact: fact(400000, '400,000', '4.3', { context: '400,000 pairs, with at most 70 %' }) },
      { label: 'Leaderboard before', fact: fact(0.9834, '0.9834', '1', { context: '0.9834 → 0.985619', split: 'leaderboard' }) },
      { label: 'Leaderboard after', fact: fact(0.985619, '0.985619', '1', { context: '0.9834 → 0.985619', split: 'leaderboard' }) },
    ],
    sections: ['4.3', '4.5'],
  },
  {
    id: 'collective',
    order: 9,
    title: 'Collective pass',
    diagram: docText('[9] Collective LightGBM (27 features) -> final p', '2.2'),
    summary:
      'A second-pass LightGBM looks at the whole candidate graph: how this business compares with rival businesses claiming the same record, and how the record resembles the business’s other confident matches.',
    models: ['lightgbm', 'qwen'],
    metrics: [
      { label: 'Features', fact: fact(27, '27', '4.2.2', { context: '| Collective input (final run) | 27 |' }) },
      { label: 'Macro F0.5, final run', fact: headline.validationF05 },
    ],
    sections: ['4.2.4'],
  },
  {
    id: 'assign',
    order: 10,
    title: 'Exclusive assignment',
    diagram: docText('[10] Exclusive assignment + p >= 0.70', '2.2'),
    summary:
      'Each record is kept only for the business that gives it the highest probability, and a match is kept only if that probability is at least 0.70.',
    models: [],
    metrics: [
      { label: 'Threshold', fact: fact(0.7, '0.70', '4.4', { context: 'p ≥ 0.70' }) },
      { label: 'Matches per business', fact: funnel.matchesTest },
      { label: 'Matched records', fact: matchingFile.matchedRecords },
    ],
    sections: ['4.4'],
    output: 'matching_results.tsv',
  },
];

// §2.2: the pairs the LLM reranker scores
export const qwenBand = docText('v3a stacker p in [0.02, 0.98]', '2.2');

export function stageById(id: StageId): PipelineStage {
  const stage = pipeline.find((s) => s.id === id);
  if (!stage) throw new Error(`Unknown pipeline stage: ${id}`);
  return stage;
}
