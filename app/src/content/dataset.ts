import { docText, fact } from './fact';

// §2.1 task and structure
export const dataset = {
  testS1: fact(1732544, '1,732,544', '2.1', { context: '1,732,544 test S1', split: 'test' }),
  testRecords: fact(9969589, '9,969,589', '2.1', { context: '9,969,589 test records', split: 'test' }),
  countriesText: docText('France, India and the US', '2.1'),
  countries: ['France', 'India', 'US'],
  singletonShareTrain: fact(5.58, '5.58 %', '2.1', { context: 'Singletons are 5.58 % of train S1', split: 'train' }),
  trueMatchesPerS1Train: fact(3.46, '3.46', '2.1', { context: 'on average 3.46 true matches', split: 'train' }),
  maxTrueMatchesTrain: fact(11, '11', '2.1', { context: '(at most 11)', split: 'train' }),
  indicShareTrain: fact(23.51, '23.51 %', '2.1', { context: 'Indic scripts: 23.51 % (train)', split: 'train' }),
  indicShareTest: fact(23.64, '23.64 %', '2.1', { context: '23.64 % (test)', split: 'test' }),
  missingAddressMin: fact(2.28, '2.28 %', '2.1', { context: '2.28 %–3.68 % of S2/S3 records' }),
  missingAddressMax: fact(3.68, '3.68 %', '2.1', { context: '2.28 %–3.68 % of S2/S3 records' }),
  sharedNameMin: fact(29.5, '29.5 %', '2.1', { context: '29.5 %–44.2 % of S1 share' }),
  sharedNameMax: fact(44.2, '44.2 %', '2.1', { context: '29.5 %–44.2 % of S1 share' }),
} as const;

// §2.2.1 distribution shift and the ghost-S1 fix
export const shift = {
  distractorsTrain: fact(26.0, '26.0 %', '2.2.1', { context: 'Train: 26.0 % of S2/S3 records match no S1', split: 'train' }),
  distractorsTestEstimate: fact(39.8, '39.8 %', '2.2.1', { context: 'Test: ≈ 39.8 % estimated', split: 'test', approx: true }),
  ghostDistractors: fact(40.79, '40.79 %', '2.2.1', { context: '40.79 % distractors' }),
  ghostS1: fact(1765456, '1,765,456', '2.2.1', { context: 'ghost train has 1,765,456 S1' }),
  ghostFraction: fact(20, '20 %', '1', { context: 'We remove 20 % of train S1' }),
  franceTestS1: fact(259452, '259,452', '2.2.1', { context: '259,452 of 1,732,544 test S1', split: 'test' }),
  franceShare: fact(14.98, '14.98 %', '2.2.1', { context: '(14.98 %)', split: 'test' }),
} as const;

// §2.1 noise types we observed
export const noiseTypes = [
  {
    noise: docText('Script and transliteration', '2.1'),
    evidence: docText('India S2 names in Indic scripts: 23.51 % (train), 23.64 % (test)', '2.1'),
  },
  {
    noise: docText('Legal suffixes and honorifics', '2.1'),
    evidence: docText('"Pvt Ltd" vs "Private Limited" vs "प्राइवेट लिमिटेड", "M/s", "Smt"', '2.1'),
  },
  {
    noise: docText('Domain-style names, phone numbers, junk', '2.1'),
    evidence: docText('"b7n.com" for "B 7 N Churchill"', '2.1'),
  },
  {
    noise: docText('Address abbreviations and reordering', '2.1'),
    evidence: docText('"Road/Rd", "Rue/R.", components in a different order', '2.1'),
  },
  {
    noise: docText('House-number changes', '2.1'),
    evidence: docText('zero-padded or ranged ("006/1", "2805-2807")', '2.1'),
  },
  {
    noise: docText('Missing addresses', '2.1'),
    evidence: docText('2.28 %–3.68 % of S2/S3 records; S1 addresses are always present', '2.1'),
  },
  {
    noise: docText('Shared names', '2.1'),
    evidence: docText(
      '29.5 %–44.2 % of S1 share their lower-cased raw name with another S1 of the same country (train and test)',
      '2.1',
    ),
  },
  { noise: docText('Unseen country', '2.1'), evidence: docText('France has no training labels', '2.1') },
] as const;
