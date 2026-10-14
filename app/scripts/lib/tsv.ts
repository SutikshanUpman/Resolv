import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

/**
 * Reads a TSV one line at a time. Fields are split on tabs only, with no quote handling,
 * matching the pipeline's reader (normalize.py uses quote_char=None because some names contain '"').
 */
export class TsvReader {
  readonly path: string;
  private readonly lines: AsyncIterator<string>;
  header: string[] = [];

  private constructor(path: string) {
    this.path = path;
    const stream = createReadStream(path, { encoding: 'utf8', highWaterMark: 1 << 20 });
    this.lines = createInterface({ input: stream, crlfDelay: Infinity })[Symbol.asyncIterator]();
  }

  static async open(path: string, expectedHeader: readonly string[]): Promise<TsvReader> {
    const reader = new TsvReader(path);
    const first = await reader.lines.next();
    reader.header = first.done ? [] : first.value.replace(/\r$/, '').split('\t');
    if (reader.header.join('\t') !== expectedHeader.join('\t')) {
      throw new Error(`${path}: expected header [${expectedHeader.join(', ')}], found [${reader.header.join(', ')}]`);
    }
    return reader;
  }

  /** The next row's fields, or null at the end of the file. Blank lines are skipped. */
  async next(): Promise<string[] | null> {
    for (;;) {
      const r = await this.lines.next();
      if (r.done) return null;
      const line = r.value.replace(/\r$/, '');
      if (line !== '') return line.split('\t');
    }
  }

  async *rows(): AsyncGenerator<string[]> {
    for (let row = await this.next(); row !== null; row = await this.next()) yield row;
  }
}

export const SOURCE_HEADER = ['entity_id', 'business_name', 'business_address', 'country'] as const;
export const CANDIDATE_HEADER = ['source1_entity_id', 'candidate_entity_ids'] as const;
export const MATCHING_HEADER = ['source1_entity_id', 'matched_entity_ids'] as const;

/**
 * Copies strings out of the file chunk they were sliced from. V8 keeps split() results as views into the
 * ~1 MB chunk readline decoded, so keeping one small row would otherwise keep its whole chunk alive.
 * Call it on anything kept beyond the current line.
 */
export function detach<T extends string | string[]>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** Splits an ID-list cell ("S2-1,S3-2" or "") into IDs. */
export function idList(cell: string | undefined): string[] {
  return cell ? cell.split(',') : [];
}
