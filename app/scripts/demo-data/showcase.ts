import { HONORIFICS, indicScript, isLegalForm, levenshteinSimilarity, nameParts, skeleton, toAscii, type IndicScript } from '../lib/text.ts';
import type { ShowcaseExample, ShowcaseRecord, SourceRow, VariantKind } from './types.ts';

/** A record word is "the same word" as an S1 word if it matches it or is a near misspelling of it. */
function sameWord(a: string, b: string): boolean {
  return a === b || (Math.min(a.length, b.length) >= 3 && levenshteinSimilarity(a, b) >= 0.6);
}

/**
 * Landing-page examples must look certain to a visitor. A record (b) qualifies only if its name is
 * clearly a variant of the S1 name (a) after the pipeline's own normalisation:
 *  - the same squashed core, or the same phonetic skeleton (this is how other scripts match);
 *  - a domain built from the name, optionally with its legal form;
 *  - otherwise every word is a word of the S1 name or a misspelling of one, and most of the name is kept.
 * Records that add words of their own (trade names, "d/b/a", extra nouns) never qualify.
 */
export function clearlySame(a: string, b: string): boolean {
  const pa = nameParts(a);
  const pb = nameParts(b);
  const sa = pa.core.join('');
  const sb = pb.core.join('');
  if (sa.length < 3 || sb.length < 3) return false;
  if (sa === sb) return true;
  const ska = skeleton(pa.core.join(' '));
  if (ska.length >= 3 && ska === skeleton(pb.core.join(' '))) return true;
  if (pb.isDomain && sb.length >= 5) {
    if (sa.includes(sb)) return true;
    if (sb.startsWith(sa) && isLegalForm(sb.slice(sa.length))) return true;
  }
  if (pb.isDomain || pb.core.some((t) => /^\d+$/.test(t))) return false;
  const explained = pb.core.every((t) => pa.core.some((s) => sameWord(s, t)));
  const kept = pa.core.filter((s) => pb.core.some((t) => sameWord(s, t))).length;
  return explained && kept >= Math.min(pa.core.length, Math.max(2, Math.ceil(pa.core.length / 2)));
}

const flat = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
const firstNumber = (s: string) => (/\d+/.exec(s)?.[0] ?? '').replace(/^0+(?=\d)/, '');

/** What differs between a record and its S1, in the noise categories of §2.1. */
export function variantKinds(s1: SourceRow, r: SourceRow): VariantKind[] {
  const kinds = new Set<VariantKind>();
  const pa = nameParts(s1.name);
  const pb = nameParts(r.name);
  const script = indicScript(r.name);
  if (script) kinds.add('script');
  if (pb.isDomain && !pa.isDomain) kinds.add('domain');
  if (r.name !== s1.name && flat(r.name) === flat(s1.name)) kinds.add('formatting');
  if (!script && !kinds.has('formatting') && pa.legalAsWritten.join(' ') !== pb.legalAsWritten.join(' ')) kinds.add('legal');
  if (pb.tokens.some((t) => HONORIFICS.has(t) && t !== 'the' && !pa.tokens.includes(t))) kinds.add('honorific');
  // A case/spacing/punctuation-only change ("E.I." for "EI") is formatting and nothing more.
  if (!script && !kinds.has('domain') && !kinds.has('formatting') && pa.core.join(' ') !== pb.core.join(' ')) {
    const setA = new Set(pa.core);
    const setB = new Set(pb.core);
    const sameSet = setA.size === setB.size && [...setA].every((t) => setB.has(t));
    if (sameSet && pa.core.length === pb.core.length) kinds.add('order');
    else if (sameSet) kinds.add('repeated');
    else if ([...setB].every((t) => setA.has(t))) kinds.add('shortened');
    else kinds.add('spelling');
  }
  if (!r.address) kinds.add('noAddress');
  else if (flat(r.address) !== flat(s1.address)) kinds.add('addressFormat');
  return [...kinds];
}

export interface ExampleRules {
  provenance: ShowcaseExample['provenance'];
  /** Require every record to carry the S1's first house number (used for test predictions). */
  sameHouseNumber: boolean;
  /** Require at least one record in an Indic script. */
  needsScript: boolean;
}

export function buildExample(s1: SourceRow, records: readonly SourceRow[], rules: ExampleRules): ShowcaseExample | null {
  if (records.length < 3 || records.length > 6 || !s1.address) return null;
  if (!records.every((r) => clearlySame(s1.name, r.name))) return null;
  if (records.filter((r) => r.name !== s1.name).length < 2) return null;
  if (rules.sameHouseNumber) {
    const n = firstNumber(s1.address);
    if (!n || !records.every((r) => r.address && firstNumber(r.address) === n)) return null;
  }
  const shown: ShowcaseRecord[] = records.map((r) => {
    const script = indicScript(r.name);
    return {
      id: r.id,
      source: r.id.startsWith('S2-') ? 'S2' : 'S3',
      name: r.name,
      address: r.address || null,
      latin: script ? toAscii(r.name) : null,
      script,
      kinds: variantKinds(s1, r),
    };
  });
  const scripts = [...new Set(shown.map((r) => r.script).filter((s): s is IndicScript => s !== null))];
  if (rules.needsScript && scripts.length === 0) return null;
  return {
    id: s1.id,
    country: s1.country,
    provenance: rules.provenance,
    s1: { id: s1.id, name: s1.name, address: s1.address },
    records: shown,
    kinds: [...new Set(shown.flatMap((r) => r.kinds))],
    scripts,
  };
}

/** Variety first: distinct noise kinds, a non-Latin script, more records. */
function score(ex: ShowcaseExample): number {
  const kinds = ex.kinds.filter((k) => k !== 'addressFormat').length;
  return kinds + (ex.scripts.length > 0 ? 2 : 0) + (ex.records.length >= 4 ? 1 : 0);
}

const LEAD_KINDS: readonly VariantKind[] = ['script', 'domain', 'honorific', 'legal', 'order', 'repeated', 'shortened', 'spelling', 'formatting'];

/**
 * Picks n examples, preferring ones that add something new (a script or a noise kind not yet shown),
 * then the highest scores. Candidates arrive in random (hash) order, so ties stay unbiased.
 */
export function pickExamples(candidates: readonly ShowcaseExample[], n: number): ShowcaseExample[] {
  const ranked = [...candidates].sort((a, b) => score(b) - score(a));
  const picked: ShowcaseExample[] = [];
  const shownKeys = new Set<string>();
  const leadKey = (ex: ShowcaseExample) => ex.scripts[0] ?? LEAD_KINDS.find((k) => ex.kinds.includes(k)) ?? 'other';
  for (const ex of ranked) {
    if (picked.length === n) break;
    if (shownKeys.has(leadKey(ex))) continue;
    picked.push(ex);
    shownKeys.add(leadKey(ex));
  }
  for (const ex of ranked) {
    if (picked.length === n) break;
    if (!picked.includes(ex)) picked.push(ex);
  }
  return picked;
}
