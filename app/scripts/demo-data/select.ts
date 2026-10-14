import { indicScript, isAcronymName } from '../lib/text.ts';
import { BUCKETS, type Bucket, type CountryStats, type PoolItem, type SourceRow } from './types.ts';

export type FlagName = 'indicMatch' | 'acronymCandidate';

type Records = ReadonlyMap<string, SourceRow>;

const FLAGS: Record<FlagName, (item: PoolItem, records: Records) => boolean> = {
  /** At least one matched record has its name in an Indic script (a cross-script match). */
  indicMatch: (item, records) => item.matches.some((id) => indicScript(records.get(id)?.name ?? '') !== null),
  /** At least one candidate record has an acronym-like name (2–5 capitals, as in §5). */
  acronymCandidate: (item, records) => item.candidates.some((id) => isAcronymName(records.get(id)?.name ?? '')),
};

export interface QuotaRule {
  flag: FlagName;
  min: number;
}

export interface CountrySelection {
  country: string;
  selected: PoolItem[];
  targets: Record<Bucket, number>;
  quota?: QuotaRule & { natural: number; final: number };
}

/**
 * Per-bucket targets for one country: fixed counts for singletons and 8+ businesses, and the rest split
 * in proportion to the country's real distribution (largest remainder, so the total is exact).
 */
export function bucketTargets(
  stats: CountryStats,
  total: number,
  fixed: Readonly<Partial<Record<Bucket, number>>>,
): Record<Bucket, number> {
  const targets = Object.fromEntries(BUCKETS.map((b) => [b, fixed[b] ?? 0])) as Record<Bucket, number>;
  const free = BUCKETS.filter((b) => fixed[b] === undefined);
  const remaining = total - Object.values(targets).reduce((a, b) => a + b, 0);
  const weight = free.reduce((sum, b) => sum + stats.buckets[b], 0);
  const shares = free.map((b) => ({ b, exact: (remaining * stats.buckets[b]) / weight }));
  for (const s of shares) targets[s.b] = Math.floor(s.exact);
  let left = remaining - shares.reduce((sum, s) => sum + Math.floor(s.exact), 0);
  for (const s of [...shares].sort((x, y) => (y.exact % 1) - (x.exact % 1))) {
    if (left-- <= 0) break;
    targets[s.b]++;
  }
  return targets;
}

/** Order in which buckets give up places to quota items: the largest, most ordinary buckets first. */
const SWAP_ORDER: readonly Bucket[] = ['3-5', '1-2', '6-7', '8+', '0'];

export function selectCountry(
  country: string,
  pools: ReadonlyMap<string, PoolItem[]>,
  targets: Record<Bucket, number>,
  records: Records,
  quota?: QuotaRule,
): CountrySelection {
  const chosen = new Map<Bucket, PoolItem[]>();
  for (const b of BUCKETS) {
    const pool = pools.get(`${country}|${b}`) ?? [];
    if (pool.length < targets[b]) throw new Error(`${country}, ${b} matches: pool has ${pool.length}, need ${targets[b]}`);
    chosen.set(b, pool.slice(0, targets[b]));
  }
  const all = () => BUCKETS.flatMap((b) => chosen.get(b) ?? []);
  if (!quota) return { country, selected: all(), targets };

  const has = (item: PoolItem) => FLAGS[quota.flag](item, records);
  const natural = all().filter(has).length;
  let count = natural;
  for (const b of SWAP_ORDER) {
    if (count >= quota.min) break;
    const picked = chosen.get(b) ?? [];
    const spare = (pools.get(`${country}|${b}`) ?? []).slice(targets[b]).filter(has);
    for (let i = picked.length - 1; i >= 0 && count < quota.min; i--) {
      const current = picked[i];
      const replacement = spare[0];
      if (!current || !replacement) break;
      if (has(current)) continue;
      picked[i] = replacement;
      spare.shift();
      count++;
    }
  }
  if (count < quota.min) throw new Error(`${country}: only ${count} businesses with ${quota.flag}, need ${quota.min}`);
  return { country, selected: all(), targets, quota: { ...quota, natural, final: count } };
}
