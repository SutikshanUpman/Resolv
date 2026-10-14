/** Deterministic hash of (seed, id) to [0, 1): FNV-1a with a murmur3 finaliser. */
export function hashUnit(seed: number, id: string): number {
  let h = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/**
 * Keeps the k items with the smallest keys seen so far. With hash keys this is a uniform random sample
 * of fixed size that does not depend on file order, so reruns give the same sample.
 */
export class BottomK<T extends { key: number }> {
  private items: T[] = [];
  private cutoff = Infinity;
  private readonly k: number;

  constructor(k: number) {
    this.k = k;
  }

  /** Whether an item with this key would be kept; lets callers skip building items that would be dropped. */
  accepts(key: number): boolean {
    return key < this.cutoff;
  }

  offer(item: T): void {
    if (item.key >= this.cutoff) return;
    this.items.push(item);
    if (this.items.length >= 2 * this.k) this.prune();
  }

  /** Items sorted by key, smallest first. */
  result(): T[] {
    this.prune();
    return this.items;
  }

  private prune(): void {
    this.items.sort((a, b) => a.key - b.key);
    if (this.items.length >= this.k) {
      this.items.length = this.k;
      this.cutoff = this.items[this.k - 1]?.key ?? Infinity;
    }
  }
}
