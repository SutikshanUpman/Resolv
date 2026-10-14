/**
 * Every number and model name on the site is wrapped in `fact()` or `docText()`.
 * `npm run check-facts` (run before every build) asserts that the quoted text appears in the named
 * section of Documentation_template.md and that a fact's quote parses to its value.
 */

/** Numbered headings of Documentation_template.md; 'title' is the header block above section 1. */
export type DocSection =
  | 'title'
  | '1'
  | '2.1'
  | '2.2'
  | '2.2.1'
  | '3.1'
  | '3.2'
  | '3.3'
  | '4.1'
  | '4.2'
  | '4.2.1'
  | '4.2.2'
  | '4.2.3'
  | '4.2.4'
  | '4.2.5'
  | '4.2.6'
  | '4.2.7'
  | '4.3'
  | '4.4'
  | '4.5'
  | '5'
  | '6'
  | 'A'
  | 'B';

/** Which data a number was measured on. Shown next to the number wherever it appears. */
export type Split = 'validation' | 'test' | 'leaderboard' | 'train' | 'oof';

export interface Fact {
  readonly kind: 'fact';
  readonly value: number;
  /** The number exactly as written in the doc, e.g. '1,732,544', '5.58 %', '~0.94', '118M'. */
  readonly quote: string;
  readonly section: DocSection;
  /** A longer passage containing `quote`, used when the bare number appears more than once in the section. */
  readonly context?: string;
  readonly split?: Split;
  /** The doc gives this value as approximate (~ or ≈). */
  readonly approx?: boolean;
  readonly unit?: '%' | 'M';
}

export interface DocText {
  readonly kind: 'doc-text';
  /** Text that appears verbatim in the section (markdown emphasis and repeated spaces ignored). */
  readonly text: string;
  readonly section: DocSection;
}

interface FactOptions {
  context?: string;
  split?: Split;
  approx?: boolean;
}

export function fact(value: number, quote: string, section: DocSection, options: FactOptions = {}): Fact {
  const unit = quote.trim().endsWith('%') ? '%' : quote.trim().endsWith('M') ? 'M' : undefined;
  return { kind: 'fact', value, quote, section, ...options, ...(unit ? { unit } : {}) };
}

export function docText(text: string, section: DocSection): DocText {
  return { kind: 'doc-text', text, section };
}
