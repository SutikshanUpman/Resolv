import { docText, fact, type DocText, type Fact } from './fact';

export type VersionId = 'v1' | 'v2' | 'v3a' | 'v3b' | 'v4a' | 'v4b';

export interface Version {
  readonly id: VersionId;
  readonly validation: Fact;
  /** Public leaderboard F0.5; v3a was never submitted. */
  readonly leaderboard?: Fact;
  readonly final?: boolean;
  readonly architecture?: DocText;
  readonly problem?: DocText;
  readonly fix?: DocText;
}

// §4.5 version history (architecture, problem, fix) and §5 runs table (validation values)
export const versions: readonly Version[] = [
  {
    id: 'v1',
    // §4.5: v1 was validated on the raw train distribution, not the ghost distribution used from v2 on
    validation: fact(0.9631, '0.96310', '4.5', { context: '0.96310 (raw distribution)', split: 'validation' }),
    leaderboard: fact(0.94, '~0.94', '4.5', { context: '| 0.96310 (raw distribution) | ~0.94 |', split: 'leaderboard', approx: true }),
    architecture: docText('8-family keys + logistic re-ranker; e5-small cross-encoder (118M); LightGBM stack', '4.5'),
    problem: docText(
      'Trained on the raw train distribution (26 % distractors; test ≈ 39.8 %); blocking ceiling only 0.9715',
      '4.5',
    ),
    fix: docText('Ghost-S1 training; dense bi-encoder; e5-base cross-encoder; LightGBM level-1 + stacker', '4.5'),
  },
  {
    id: 'v2',
    validation: fact(0.99013, '0.99013', '4.5', { context: '| 0.99013 | 0.981874 |', split: 'validation' }),
    leaderboard: fact(0.981874, '0.981874', '4.5', { context: '| 0.99013 | 0.981874 |', split: 'leaderboard' }),
    architecture: docText(
      '11-family keys ∪ e5-small bi-encoder (118M); 39-feature LightGBM re-ranker; level-1 LightGBM; e5-base CE (278M); LightGBM stacker',
      '4.5',
    ),
    problem: docText(
      '2,346 true val pairs lost at the stage-2 cut (mostly address-less records with shared names, and S1 with > 8 true records); val→LB gap; France unseen',
      '4.5',
    ),
    fix: docText('Two-pass re-ranker; level-1 on all rows; CE record-side features; collective pass', '4.5'),
  },
  {
    id: 'v3a',
    validation: fact(0.99053, '0.99053', '5', { context: '| v3a | 0.99053 |', split: 'validation' }),
  },
  {
    id: 'v3b',
    validation: fact(0.99098, '0.99098', '4.5', { context: '0.99098 (v3b)', split: 'validation' }),
    leaderboard: fact(0.9834, '0.9834', '4.5', { context: '0.9834 (v3b)', split: 'leaderboard' }),
    architecture: docText('v2 + two-pass re-ranker, level-1 on all rows, CE record-side features, collective LightGBM', '4.5'),
    problem: docText(
      'Acronyms, French abbreviations and region names, coined trade names: cases the e5-base cross-encoder handles poorly',
      '4.5',
    ),
    fix: docText('Qwen3-Reranker-0.6B on uncertain pairs', '4.5'),
  },
  {
    id: 'v4a',
    validation: fact(0.99153, '0.99153', '4.5', { context: '0.99153 (v4a)', split: 'validation' }),
    leaderboard: fact(0.985619, '0.985619', '4.5', { context: '0.985619 (v4a)', split: 'leaderboard' }),
    architecture: docText('v3 + Qwen3-Reranker-0.6B (596M) as a collective feature', '4.5'),
    problem: docText('Qwen coverage limited to [0.05, 0.95] (v4a)', '4.5'),
    fix: docText('v4b widened the band to [0.02, 0.98]', '4.5'),
  },
  {
    id: 'v4b',
    final: true,
    validation: fact(0.99158, '0.99158', '4.5', { context: '0.99158 (v4b)', split: 'validation' }),
    leaderboard: fact(0.985978, '0.985978', '4.5', { context: '0.985978 (v4b)', split: 'leaderboard' }),
  },
];

// §4.5 the story behind each leaderboard step
export const versionSteps = [
  {
    from: 'v1',
    to: 'v2',
    text: docText(
      'v1 scored 0.96310 on a validation split drawn from train, but about 0.94 on the leaderboard. The test set has more S2/S3 records per S1 (5.53–5.82 against 4.67–4.68 in train), so a model tuned on train over-predicts on test. v2 therefore trains and tunes everything on the ghost-S1 distribution, and adds a dense retriever that lifts the blocking ceiling from 0.9715 to 0.99792.',
      '4.5',
    ),
  },
  {
    from: 'v2',
    to: 'v3b',
    text: docText(
      'The v2 cut discarded 2,346 true validation pairs. These were mostly address-less records whose name several S1 share, and records of S1 with more than 8 true matches. Such records look weak to every claimant when scored alone.',
      '4.5',
    ),
  },
  {
    from: 'v3b',
    to: 'v4a',
    text: docText(
      'A multilingual LLM reranker, applied only where the stacker is uncertain, brings that knowledge at a manageable cost. It gave the largest leaderboard step, +0.0022 (0.9834 → 0.985619).',
      '4.5',
    ),
  },
  {
    from: 'v4a',
    to: 'v4b',
    text: docText('Widening the scored band in v4b added a further +0.000359 (0.985978).', '4.5'),
  },
] as const;

// §5 Stage B: the run we did not select, and why
export const stageB = {
  validation: fact(0.9916, '0.9916', '5', { context: '| Stage B (not selected) | 0.9916 |', split: 'validation' }),
  singletons: fact(0.99463, '0.99463', '5', { context: '(0.99463 vs 0.99517)', split: 'validation' }),
  reason: docText(
    'Its validation F0.5 was 0.00002 higher, but its singleton score was lower (0.99463 vs 0.99517), so we kept v4b.',
    '5',
  ),
} as const;
