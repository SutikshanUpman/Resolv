import { funnel } from './blocking';
import { dataset } from './dataset';
import { decision } from './decision';
import { docText, type DocText, type Fact } from './fact';
import type { StageId } from './pipeline';

export interface PhaseStep {
  readonly label: string;
  /** Steps that run side by side are drawn in one row. */
  readonly parallel?: readonly string[];
}

export interface OverviewPhase {
  readonly id: 'find' | 'score' | 'decide';
  readonly title: string;
  readonly stages: readonly StageId[];
  readonly summary: string;
  readonly steps: readonly PhaseStep[];
  /** What leaves this phase, per business. */
  readonly result: { readonly label: string; readonly fact?: Fact; readonly also?: Fact };
}

// §1 summary and §2.2: the ten stages fall into three phases.
export const overview = {
  lead: 'First we use cheap methods to narrow down the search. Then stronger models judge the short list, and a final rule makes the decision.',
  input: {
    title: 'In',
    s1: 'One reference business (S1)',
    sources: 'Two more sources of business records (S2, S3)',
    s1Count: dataset.testS1,
    recordCount: dataset.testRecords,
  },
  output: {
    title: 'Out',
    rule: docText('For each S1 we output the list of S2/S3 ids that describe the same business, possibly empty.', '2.1'),
  },
  phases: [
    {
      id: 'find',
      title: 'Find candidates',
      stages: ['normalise', 'retrieve', 'union', 'rerankCut'],
      summary:
        'Clean every name and address, let two different search methods pull in every record that could be the business, then keep only a short list.',
      steps: [
        { label: 'Clean and transliterate' },
        { label: 'Search', parallel: ['Shared keywords', 'Embedding model'] },
        { label: 'Merge both lists' },
        { label: 'Fast re-ranker keeps a short list' },
      ],
      result: { label: 'candidates per business', fact: funnel.afterCutValidation, also: funnel.afterCutTest },
    },
    {
      id: 'score',
      title: 'Score each pair',
      stages: ['pairFeatures', 'crossEncoder', 'stacker', 'llmRerank', 'collective'],
      summary:
        'Several models judge every business–record pair on the short list. Each adds evidence the others miss, and their scores are combined into one probability.',
      steps: [
        { label: 'Judge each pair', parallel: ['Similarity features + trees', 'Cross-encoder'] },
        { label: 'Stacker combines the scores' },
        { label: 'LLM re-checks unsure pairs only' },
        { label: 'Compare with rival businesses' },
      ],
      result: { label: 'one match probability per pair' },
    },
    {
      id: 'decide',
      title: 'Decide',
      stages: ['assign'],
      summary:
        'A record can belong to only one business, so each record goes to the business that wants it most, and only confident matches are kept.',
      steps: [{ label: 'One record, one business' }, { label: `Keep only if p ≥ ${decision.threshold.quote}` }],
      result: { label: 'matches per business', fact: funnel.matchesTest },
    },
  ] satisfies readonly OverviewPhase[],
} as const;

// Short station names for the pipeline route map.
export const stageShortTitles: Readonly<Record<StageId, string>> = {
  normalise: 'Clean',
  retrieve: 'Two searches',
  union: 'Merge',
  rerankCut: 'Re-rank & cut',
  pairFeatures: 'Pair features',
  crossEncoder: 'Cross-encoder',
  stacker: 'Stacker',
  llmRerank: 'LLM reranker',
  collective: 'Collective',
  assign: 'Assign',
};

// One plain line per stage, shown under the stage title in the pipeline detail card.
export const stageOneLiners: Readonly<Record<StageId, string>> = {
  normalise: 'Clean every name and address and transliterate Indic scripts.',
  retrieve: 'Two search methods run side by side: shared keywords and a multilingual embedding model.',
  union: 'Merge the two lists into one candidate set.',
  rerankCut: 'A fast model ranks the candidates and keeps a short list.',
  pairFeatures: 'Measure how alike each pair is, then a tree model gives a first match probability.',
  crossEncoder: 'A transformer reads both records side by side.',
  stacker: 'Combine every score so far into one probability.',
  llmRerank: 'A small LLM re-checks only the pairs the models are unsure about.',
  collective: 'Compare each match with rival businesses and with the business’s other matches.',
  assign: 'Give each record to at most one business, and keep only confident matches.',
};

export interface GlossaryTerm {
  readonly term: string;
  readonly definition: string | DocText;
}

// Terms used on the architecture page. Definitions quoted from the doc where it gives one.
export const glossary: readonly GlossaryTerm[] = [
  {
    term: 'S1, S2, S3',
    definition: 'S1 is the reference table of businesses. S2 and S3 are two other sources whose records we link to S1.',
  },
  {
    term: 'Candidate',
    definition: 'A record from S2 or S3 that might be the same business as an S1, kept for the matching models to judge.',
  },
  { term: 'Pair recall', definition: docText('Pair recall is the share of true (S1, record) pairs that are in the candidate set.', '3.1') },
  {
    term: 'Oracle F0.5',
    definition: docText(
      'Oracle F0.5 is the macro F0.5 that a perfect matcher would reach when restricted to these candidates: the ceiling that blocking leaves for the matcher.',
      '3.1',
    ),
  },
  {
    term: 'Bi-encoder',
    definition: 'Turns each record into a vector on its own, so millions of records can be searched quickly by similarity. Used to find candidates.',
  },
  {
    term: 'Cross-encoder',
    definition: 'Reads two records together in one pass. Slower but more accurate than a bi-encoder, so it only scores the short list.',
  },
  {
    term: 'LightGBM',
    definition: 'Gradient-boosted decision trees: a fast model that works well on hand-built features such as name and address similarity.',
  },
  {
    term: 'Level-1 model and stacker',
    definition: 'The level-1 model scores pairs from the hand-built features. The stacker is a second model that combines the features, the level-1 scores and the cross-encoder’s scores into one probability.',
  },
  {
    term: 'Competition features',
    definition: 'Features that show how strongly other businesses claim the same record. They put the “one record, one business” rule into every learned stage.',
  },
  {
    term: 'Macro F0.5',
    definition: 'The challenge score: F0.5 is computed for each business, then averaged. It weighs precision about twice as much as recall.',
  },
];
