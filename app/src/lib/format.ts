import type { Fact } from '../content/fact';

/** Number of decimals the doc writes for this fact ("0.99158" → 5, "1,732,544" → 0). */
export function quoteDecimals(f: Fact): number {
  const n = f.quote.replace(/[,~≈%M\s]/g, '');
  const dot = n.indexOf('.');
  return dot < 0 ? 0 : n.length - dot - 1;
}

interface FormatOptions {
  /** Show fewer decimals than the doc. */
  digits?: number;
  /**
   * 'down' (default) never shows a value larger than the doc's. 'nearest' is allowed only where
   * CLAUDE.md permits it: the leaderboard 0.985978 may be shown as 0.986.
   */
  rounding?: 'down' | 'nearest';
}

/** Formats a fact as the doc writes it: same decimals, thousands separators, ~ for approximate values. */
export function formatFact(f: Fact, { digits, rounding = 'down' }: FormatOptions = {}): string {
  const d = digits ?? quoteDecimals(f);
  const scale = 10 ** d;
  const value = rounding === 'down' ? Math.floor(f.value * scale + 1e-9) / scale : f.value;
  const text = new Intl.NumberFormat('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }).format(value);
  const unit = f.unit === '%' ? '%' : f.unit === 'M' ? 'M' : '';
  return `${f.approx ? '~' : ''}${text}${unit}`;
}
