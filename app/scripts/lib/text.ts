import anyAscii from 'any-ascii';

/**
 * A port of the name rules in normalize.py (NAME_PRE, DOMAIN_RX, TOKEN_CANON, DROP_NAME, skeleton).
 * Used only by build-demo-data to pick landing-page examples whose variants are clearly the same name.
 */

const INDIC_SCRIPTS = [
  'Devanagari',
  'Bengali',
  'Gujarati',
  'Gurmukhi',
  'Oriya',
  'Tamil',
  'Telugu',
  'Kannada',
  'Malayalam',
] as const;
export type IndicScript = (typeof INDIC_SCRIPTS)[number];

const SCRIPT_RES = INDIC_SCRIPTS.map((s) => [s, new RegExp(`\\p{Script=${s}}`, 'u')] as const);

/** The first Indic script found in the text (the same nine scripts as normalize.py's INDIC). */
export function indicScript(text: string): IndicScript | null {
  for (const [script, re] of SCRIPT_RES) if (re.test(text)) return script;
  return null;
}

/** "Acronym-like record names (2–5 capitals)", the definition in §5 of the doc. */
export function isAcronymName(name: string): boolean {
  return /^[A-Z]{2,5}$/.test(name.trim());
}

const LEGAL_CANON: Record<string, string> = {
  private: 'pvt', pvt: 'pvt', pte: 'pvt', pvtltd: 'pvt', limited: 'ltd', ltd: 'ltd', ltda: 'ltd',
  incorporated: 'inc', inc: 'inc', corporation: 'corp', corp: 'corp', corpn: 'corp',
  company: 'co', co: 'co', cie: 'co', llc: 'llc', llp: 'llp', lp: 'lp', plc: 'plc',
  sarl: 'sarl', sas: 'sas', sasu: 'sas', sa: 'sa', eurl: 'eurl', sci: 'sci', snc: 'snc',
  selarl: 'selarl', scop: 'scop', scp: 'scp', gie: 'gie', earl: 'earl', gaec: 'gaec', sccv: 'sccv',
  gmbh: 'gmbh', pty: 'pty', opc: 'opc', pc: 'pc', pllc: 'pllc', ltee: 'ltd',
  praivet: 'pvt', praaivet: 'pvt', praivett: 'pvt', prayvet: 'pvt', praibhet: 'pvt', praivhet: 'pvt',
  limitedd: 'ltd', limitad: 'ltd', 'limited.': 'ltd', 'limitedd.': 'ltd', limitett: 'ltd', limitted: 'ltd',
  elelpi: 'llp', elelpii: 'llp', kampani: 'co', kompani: 'co', inka: 'inc',
};
const TOKEN_CANON: Record<string, string> = { ...LEGAL_CANON, ets: 'etablissements', etabl: 'etablissements', et: 'and' };
const LEGAL_VALS = new Set(Object.values(LEGAL_CANON));
export const HONORIFICS = new Set(['smt', 'sri', 'shri', 'shree', 'sh', 'mr', 'mrs', 'ms', 'dr', 'kumari', 'messrs', 'the']);

const NAME_PRE: readonly (readonly [RegExp, string])[] = [
  [/\+?\d[\d\s-]{7,}\d/g, ' '],
  [/\bm\s*\/\s*s\b\.?/g, ' '],
  [/\bl\.\s*l\.\s*c\b\.?/g, ' llc '],
  [/\bl\.\s*l\.\s*p\b\.?/g, ' llp '],
  [/\bs\.\s*a\.\s*r\.\s*l\b\.?/g, ' sarl '],
  [/\be\.\s*u\.\s*r\.\s*l\b\.?/g, ' eurl '],
  [/\bs\.\s*a\.\s*s\.\s*u\b\.?/g, ' sasu '],
  [/\bs\.\s*a\.\s*s\b\.?/g, ' sas '],
  [/\bs\.\s*n\.\s*c\b\.?/g, ' snc '],
  [/\bs\.\s*c\.\s*i\b\.?/g, ' sci '],
  [/\bs\.\s*a\.(\s|$)/g, ' sa '],
  [/\bp\.?\s*ltd\b/g, ' pvt ltd '],
  [/&/g, ' and '],
  [/\+/g, ' and '],
];
const DOMAIN_RE = /\b(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:co\.in|com|in|net|org|co|fr|biz|info|us)\b/g;

/** anyascii transliteration of NFKC text, as in normalize.py's _to_ascii. */
export function toAscii(text: string): string {
  return anyAscii(text.normalize('NFKC'));
}

export interface NameParts {
  /** Canonical tokens (normalize.py's name_norm). */
  tokens: string[];
  /** Tokens without legal forms and honorifics (name_core). */
  core: string[];
  legal: string[];
  /** Legal-form words as written, before canonicalisation ("private limited", "pvt ltd"). */
  legalAsWritten: string[];
  isDomain: boolean;
}

export function nameParts(name: string): NameParts {
  let s = toAscii(name).toLowerCase();
  const legalAsWritten = s
    .split(/[^a-z0-9.]+/)
    .map((t) => t.replace(/\.+$/, ''))
    .filter((t) => LEGAL_VALS.has(LEGAL_CANON[t] ?? ''));
  for (const [re, rep] of NAME_PRE) s = s.replace(re, rep);
  const isDomain = new RegExp(DOMAIN_RE.source).test(s);
  s = s.replace(DOMAIN_RE, ' $1 ').replace(/[^a-z0-9]+/g, ' ').trim();
  const tokens = s.split(' ').filter(Boolean).map((t) => TOKEN_CANON[t] ?? t);
  const kept = tokens.filter((t) => !LEGAL_VALS.has(t) && !HONORIFICS.has(t));
  const legal = [...new Set(tokens.filter((t) => LEGAL_VALS.has(t)))].sort();
  return { tokens, core: kept.length > 0 ? kept : tokens, legal, legalAsWritten, isDomain };
}

const SK_RULES: readonly (readonly [RegExp, string])[] = [
  [/[^a-z ]/g, ''],
  [/tion/g, 'shn'],
  [/sion/g, 'shn'],
  [/ph/g, 'f'],
  [/gh/g, 'g'],
  [/kh/g, 'k'],
  [/bh/g, 'b'],
  [/dh/g, 'd'],
  [/th/g, 't'],
  [/sh/g, 's'],
  [/ch/g, 'k'],
  [/ck/g, 'k'],
  [/x/g, 'ks'],
  [/q/g, 'k'],
  [/c(?=[eiy])/g, 's'],
  [/c/g, 'k'],
  [/z/g, 's'],
  [/g(?=[eiy])/g, 's'],
  [/j/g, 's'],
  [/m(?=[^aeiouy ])/g, 'n'],
];

function skeletonToken(t: string): string {
  if (!t) return '';
  const first = t[0] === 'w' ? 'v' : t.charAt(0);
  const rest = t.slice(1).replace(/[aeiouywhv]/g, '');
  return (first + rest).replace(/(.)\1+/g, '$1');
}

/** Phonetic consonant skeleton of a name_core string (normalize.py's skeleton). */
export function skeleton(nameCore: string): string {
  let s = nameCore;
  for (const [re, rep] of SK_RULES) s = s.replace(re, rep);
  return s
    .split(' ')
    .map(skeletonToken)
    .filter(Boolean)
    .sort()
    .join(' ');
}

function bigrams(s: string): Map<string, number> {
  const m = new Map<string, number>();
  for (let i = 0; i < s.length - 1; i++) {
    const g = s.slice(i, i + 2);
    m.set(g, (m.get(g) ?? 0) + 1);
  }
  return m;
}

export function isLegalForm(token: string): boolean {
  return LEGAL_VALS.has(token);
}

/** 1 − edit distance / longer length, 0 to 1. */
export function levenshteinSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0] ?? 0;
    prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const up = prev[j] ?? 0;
      prev[j] = Math.min(up + 1, (prev[j - 1] ?? 0) + 1, diag + (a[i - 1] === b[j - 1] ? 0 : 1));
      diag = up;
    }
  }
  return 1 - (prev[b.length] ?? 0) / Math.max(a.length, b.length);
}

/** Sørensen–Dice similarity of character bigrams, 0 to 1. */
export function dice(a: string, b: string): number {
  if (a === b) return 1;
  if (a.length < 2 || b.length < 2) return 0;
  const ga = bigrams(a);
  const gb = bigrams(b);
  let overlap = 0;
  for (const [g, n] of ga) overlap += Math.min(n, gb.get(g) ?? 0);
  return (2 * overlap) / (a.length - 1 + (b.length - 1));
}
