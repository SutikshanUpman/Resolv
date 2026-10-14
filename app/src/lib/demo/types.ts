/** Shapes of the files in public/demo/, written by scripts/build-demo-data.ts. */

export type VariantKind =
  | 'script'
  | 'domain'
  | 'legal'
  | 'honorific'
  | 'formatting'
  | 'order'
  | 'spelling'
  | 'shortened'
  | 'repeated'
  | 'noAddress'
  | 'addressFormat';

export interface ShowcaseRecord {
  id: string;
  source: 'S2' | 'S3';
  name: string;
  address: string | null;
  /** anyascii transliteration, only for names in an Indic script. */
  latin: string | null;
  script: string | null;
  kinds: VariantKind[];
}

export interface ShowcaseExample {
  id: string;
  country: string;
  /** 'verified': train ground truth. 'pipeline': our test predictions (the test set has no labels). */
  provenance: 'verified' | 'pipeline';
  s1: { id: string; name: string; address: string };
  records: ShowcaseRecord[];
  kinds: VariantKind[];
  scripts: string[];
}

export interface ShowcaseFile {
  description: string;
  examples: ShowcaseExample[];
}
