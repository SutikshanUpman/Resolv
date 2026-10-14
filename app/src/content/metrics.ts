import { fact } from './fact';

// §1 headline result (final submission, v4b)
export const headline = {
  leaderboardF05: fact(0.985978, '0.985978', '1', { split: 'leaderboard' }),
  validationF05: fact(0.99158, '0.99158', '1', { split: 'validation' }),
  validationPrecision: fact(0.99803, '0.99803', '1', { split: 'validation' }),
  validationRecall: fact(0.97776, '0.97776', '1', { split: 'validation' }),
  singletonF05: fact(0.99517, '0.99517', '1', { split: 'validation' }),
  // §1: size of the held-out validation set behind every "validation" number
  validationS1: fact(100000, '100,000', '1', { context: '100,000 held-out train S1' }),
} as const;
