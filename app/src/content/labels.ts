import type { Split } from './fact';

// How each data split is named next to a number (§1, §2.2.1, §4.3 describe the splits).
export const splitLabels: Readonly<Record<Split, { short: string; long: string }>> = {
  validation: { short: 'Validation', long: 'Validation: 100,000 held-out train S1 with a test-like share of unmatched records' },
  test: { short: 'Test set', long: 'Challenge test set, no labels' },
  leaderboard: { short: 'Public leaderboard', long: 'Public leaderboard, from our submission history' },
  train: { short: 'Train', long: 'Challenge training set' },
  oof: { short: 'Out-of-fold', long: 'Out-of-fold predictions on stack_train' },
};
