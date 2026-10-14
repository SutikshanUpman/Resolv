// Product identity. Components read the name from here only; never hard-code it.
export const brand = {
  name: 'Resolv',
  tagline: 'One business, however it’s written.',
  // Landing hero headline. The tagline stays the short line for the footer and page title.
  hook: 'Still matching business records by hand?',
  subline:
    'Resolv finds every record of the same business across three sources, in English, French and Indic scripts.',
  description:
    'Business entity resolution across three sources and three countries, built for the Amazon ML Challenge 2026 by team The Epoch Warriors.',
  event: 'Amazon ML Challenge 2026',
} as const;
