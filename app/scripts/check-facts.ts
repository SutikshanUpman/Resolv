/**
 * Verifies every fact in src/content against Documentation_template.md:
 *  - the quoted text (and its context, if any) appears in the named section;
 *  - the quote parses to the stored value, and approximate values are marked as such;
 *  - no bare number sits in the content modules outside a fact (except structural keys).
 * Runs before every build; any failure stops the build.
 */
import { readFileSync } from 'node:fs';
import * as content from '../src/content/index.ts';
import type { DocText, Fact } from '../src/content/index.ts';
import { DOC_PATH } from './lib/paths.ts';

/** Keys whose numbers are structure, not claims: list positions, table headers (n), the metric's β. */
const STRUCTURAL_KEYS = new Set(['order', 'n', 'beta']);

function normalise(text: string): string {
  return text.replace(/[*`]/g, '').replace(/\s+/g, ' ').trim();
}

function parseSections(markdown: string): Map<string, string> {
  const lines = markdown.split(/\r?\n/);
  const heads: { id: string; level: number; line: number }[] = [];
  lines.forEach((line, i) => {
    const m = /^(#{2,4})\s+(\d+(?:\.\d+)*|[A-Z])\.?\s/.exec(line);
    if (m?.[1] && m[2]) heads.push({ id: m[2], level: m[1].length, line: i });
  });
  const sections = new Map<string, string>();
  const firstHeading = lines.findIndex((l) => l.startsWith('## '));
  sections.set('title', normalise(lines.slice(0, firstHeading).join('\n')));
  heads.forEach((h, i) => {
    const end = heads.slice(i + 1).find((n) => n.level <= h.level)?.line ?? lines.length;
    sections.set(h.id, normalise(lines.slice(h.line, end).join('\n')));
  });
  return sections;
}

function isFact(v: unknown): v is Fact {
  return typeof v === 'object' && v !== null && (v as { kind?: unknown }).kind === 'fact';
}

function isDocText(v: unknown): v is DocText {
  return typeof v === 'object' && v !== null && (v as { kind?: unknown }).kind === 'doc-text';
}

const sections = parseSections(readFileSync(DOC_PATH, 'utf8'));
const errors: string[] = [];
const seen = new Set<object>();
let facts = 0;
let texts = 0;

function sectionText(id: string, where: string): string | null {
  const text = sections.get(id);
  if (text === undefined) errors.push(`${where}: section §${id} not found in the doc`);
  return text ?? null;
}

function checkFact(f: Fact, where: string): void {
  facts++;
  const section = sectionText(f.section, where);
  if (section === null) return;
  const quote = normalise(f.quote);
  const context = normalise(f.context ?? f.quote);
  if (!context.includes(quote)) errors.push(`${where}: context "${context}" does not contain quote "${quote}"`);
  if (!section.includes(context)) errors.push(`${where}: "${context}" not found in §${f.section}`);

  const numeric = quote.replace(/[,~≈%M\s]/g, '');
  if (!/^-?\d+(\.\d+)?$/.test(numeric)) {
    errors.push(`${where}: quote "${quote}" is not a single number`);
  } else if (Math.abs(Number(numeric) - f.value) > 1e-12) {
    errors.push(`${where}: quote "${quote}" parses to ${numeric}, but the value is ${f.value}`);
  }
  const docSaysApprox = /[~≈]/.test(quote) || new RegExp(`[~≈]\\s*${quote.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`).test(context);
  if (docSaysApprox && !f.approx) errors.push(`${where}: the doc gives "${quote}" as approximate; mark it approx`);
  if (f.approx && !docSaysApprox) errors.push(`${where}: marked approx, but the doc gives "${quote}" as exact`);
}

function checkText(t: DocText, where: string): void {
  texts++;
  const section = sectionText(t.section, where);
  if (section !== null && !section.includes(normalise(t.text))) {
    errors.push(`${where}: text not found in §${t.section}: "${t.text}"`);
  }
}

function walk(value: unknown, path: string, key: string): void {
  if (typeof value === 'number') {
    if (!STRUCTURAL_KEYS.has(key)) errors.push(`${path}: bare number ${value}; wrap it in fact() with its doc quote`);
    return;
  }
  if (typeof value !== 'object' || value === null || typeof value === 'function') return;
  if (seen.has(value)) return;
  seen.add(value);
  if (isFact(value)) return checkFact(value, path);
  if (isDocText(value)) return checkText(value, path);
  for (const [k, v] of Object.entries(value)) walk(v, `${path}.${k}`, k);
}

for (const [name, value] of Object.entries(content)) walk(value, name, name);

if (errors.length > 0) {
  console.error(`check-facts: ${errors.length} problem(s)\n`);
  for (const e of errors) console.error(`  ✗ ${e}`);
  process.exit(1);
}
console.log(`check-facts: ${facts} numbers and ${texts} text passages match Documentation_template.md`);
