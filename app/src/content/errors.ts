import { docText, fact } from './fact';

// §5 where the remaining F0.5 is lost (final run, validation)
export const lossByType = [
  { label: docText('true records missed (no wrong records)', '5'), points: fact(0.00527, '0.00527', '5', { split: 'validation' }) },
  { label: docText('S1 with true records predicted empty', '5'), points: fact(0.00151, '0.00151', '5', { split: 'validation' }) },
  { label: docText('extra wrong record (no misses)', '5'), points: fact(0.00123, '0.00123', '5', { split: 'validation' }) },
  { label: docText('singleton given a match', '5'), points: fact(0.00027, '0.00027', '5', { split: 'validation' }) },
  { label: docText('both missed and wrong records', '5'), points: fact(0.00014, '0.00014', '5', { split: 'validation' }) },
] as const;

export const lossTotal = fact(0.00842, '0.00842', '5', { context: 'total (= 1 − 0.99158) | 0.00842', split: 'validation' });

// §5 where true validation pairs are lost, v2 against the final pipeline
export const truePairsValidation = fact(346212, '346,212', '5', { split: 'validation' });

export const lossByStage = [
  {
    stage: docText('never retrieved by sparse or dense', '5'),
    v2: fact(488, '488', '5', { context: '| never retrieved by sparse or dense | 488 |', split: 'validation' }),
    final: fact(488, '488', '5', { context: '| never retrieved by sparse or dense | 488 | 488 |', split: 'validation' }),
  },
  {
    stage: docText('removed by the stage-2 cut', '5'),
    v2: fact(2346, '2,346', '5', { context: '| removed by the stage-2 cut | 2,346 |', split: 'validation' }),
    final: fact(1492, '1,492', '5', { split: 'validation' }),
  },
  {
    stage: docText('rejected by the matcher', '5'),
    v2: fact(6150, '6,150', '5', { split: 'validation' }),
    final: fact(5864, '5,864', '5', { split: 'validation' }),
  },
  {
    stage: docText('found', '5'),
    v2: fact(337228, '337,228', '5', { split: 'validation' }),
    final: fact(338368, '338,368', '5', { split: 'validation' }),
  },
] as const;

// §5 logged diagnostics on the address-less, shared-name cases
export const diagnostics = {
  addresslessLabel: docText('precision of address-less candidate pairs with 0.40 ≤ p < 0.70', '5'),
  addresslessPrecision: fact(0.52, '0.52', '5', { context: '0.52 (below the 0.698 break-even)', split: 'validation' }),
  emptyAreSingletons: fact(97.4, '97.4 %', '5', { split: 'validation' }),
  identicalSpellingMisses: fact(57.5, '57.5 %', '5', { split: 'validation' }),
  siblingHelpful: fact(17, '17 %', '5', { context: '17 % vs 8 %', split: 'validation' }),
  siblingMisleading: fact(8, '8 %', '5', { context: '17 % vs 8 %', split: 'validation' }),
  conclusion: docText(
    'Rules that force a match for every S1 would hurt, because almost all empty predictions are true singletons. We therefore abstain on them.',
    '5',
  ),
} as const;

// §5 France has no labels; these are observations on test
export const countryObservations = [
  {
    country: 'France',
    matchesPerS1: fact(3.321, '3.321', '5', { split: 'test' }),
    emptyShare: fact(5.75, '5.75 %', '5', { split: 'test' }),
    acronymShare: fact(1.67, '1.67 %', '5', { split: 'test' }),
  },
  {
    country: 'India',
    matchesPerS1: fact(3.39, '3.39', '5', { context: 'India (3.39, 5.76 %)', split: 'test' }),
    emptyShare: fact(5.76, '5.76 %', '5', { split: 'test' }),
    acronymShare: fact(0.19, '0.19 %', '5', { split: 'test' }),
  },
  {
    country: 'US',
    matchesPerS1: fact(3.385, '3.385', '5', { split: 'test' }),
    emptyShare: fact(5.82, '5.82 %', '5', { split: 'test' }),
    acronymShare: fact(0.09, '0.09 %', '5', { split: 'test' }),
  },
] as const;

export const france = {
  departementShare: fact(31.84, '31.84 %', '5', { split: 'test' }),
  regionShare: fact(33.29, '33.29 %', '5', { split: 'test' }),
  acronymMatchedBefore: fact(72, '72 %', '5', { context: 'rose from 72 % to 84 %' }),
  acronymMatchedAfter: fact(84, '84 %', '5', { context: 'rose from 72 % to 84 %' }),
  acronymNote: docText('values logged from our training runs', '5'),
  gap: docText(
    'The gap between validation (0.99158) and public LB (0.985978) is expected: France (~15 % of test S1) has no labels and cannot be validated, the test distractor rate is estimated rather than known, and the public LB covers only a subset of test.',
    '5',
  ),
} as const;

// §5 limitations, stated as the doc states them
export const limitations = [
  {
    title: docText('Address-less records with shared names', '5'),
    text: docText('remain the main error source, and the text holds no signal to separate them.', '5'),
  },
  {
    title: docText('France is unvalidated.', '5'),
    text: docText('No label exists, and the leaderboard gives only one aggregate score.', '5'),
  },
  {
    title: docText('Transformer training is not bit-exact.', '5'),
    text: docText(
      'A re-run reproduces the results within run-to-run noise, not to the last digit.',
      '5',
    ),
  },
] as const;
