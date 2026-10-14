import type { VariantKind } from '../lib/demo/types';

// Labels for what differs between a record and its S1, in the noise categories of §2.1.
export const variantLabels: Readonly<Record<VariantKind, string>> = {
  script: 'Other script',
  domain: 'Domain-style name',
  legal: 'Legal form',
  honorific: 'Honorific',
  formatting: 'Case or punctuation',
  order: 'Word order',
  spelling: 'Typo',
  shortened: 'Words dropped',
  repeated: 'Repeated word',
  noAddress: 'No address',
  addressFormat: 'Address format',
};

export const provenanceLabels = {
  verified: { short: 'Verified', long: 'Verified: a true match in the challenge’s training labels' },
  pipeline: { short: 'Matched by our pipeline', long: 'Matched by our pipeline on the test set, which has no labels' },
} as const;
