import { docText, fact } from './fact';

// F0.5 per business, then the macro average over all S1 (challenge README, "Evaluation Criteria"):
//   F0.5 = (1.25 × P × R) / (0.25 × P + R)
export const fBeta = {
  beta: 0.5,
  formula: 'F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)',
  precisionWeight: docText('F0.5 weighs precision about twice as much as recall.', '2.1'),
  singletonRule: docText(
    'An S1 with no true match (a singleton) scores 1 only if we predict an empty list, and 0 otherwise.',
    '2.1',
  ),
  singletonCost: docText('On a true singleton, a wrong record costs the full 1.0.', '4.4'),
} as const;

// §4.4 decision rule
export const decision = {
  threshold: fact(0.7, '0.70', '4.4', { context: 'p ≥ 0.70' }),
  v2OofThreshold: fact(0.69, '0.69', '4.4', { context: 'the OOF-optimal threshold on a 0.01 grid was 0.69', split: 'oof' }),
  exclusive: docText(
    'Each S2/S3 record is kept only for the S1 that gives it the highest final probability.',
    '4.4',
  ),
} as const;

// §4.4 break-even precision: an S1 with n true records, n − 1 already predicted
export const breakEven = [
  { n: 2, precision: fact(0.632, '0.632', '4.4') },
  { n: 3, precision: fact(0.698, '0.698', '4.4', { context: '| 0.632 | 0.698 |' }) },
  { n: 4, precision: fact(0.727, '0.727', '4.4') },
  { n: 5, precision: fact(0.743, '0.743', '4.4') },
] as const;
