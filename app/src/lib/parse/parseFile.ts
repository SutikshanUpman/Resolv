/** A parsed table: header row plus data rows, all as strings. */
export interface ParsedTable {
  fileName: string;
  size: number;
  format: 'TSV' | 'CSV' | 'XLSX';
  columns: string[];
  rows: string[][];
}

/** Largest file parsed in the browser. */
export const MAX_BYTES = 30 * 1024 * 1024;

export class ParseError extends Error {}

function formatOf(name: string): ParsedTable['format'] {
  const ext = name.toLowerCase().split('.').pop() ?? '';
  if (ext === 'xlsx' || ext === 'xls') return 'XLSX';
  if (ext === 'csv') return 'CSV';
  if (ext === 'tsv' || ext === 'txt') return 'TSV';
  throw new ParseError(`“${name}” isn’t a CSV, TSV or Excel file.`);
}

/**
 * Tab-separated text, split on tabs only. Like the challenge pipeline (quote_char=None), quotes are kept as
 * text, because some business names contain a literal ".
 */
function parseTsv(text: string): string[][] {
  return text
    .replace(/^\uFEFF/, '')
    .split(/\r?\n/)
    .filter((l) => l.trim() !== '')
    .map((l) => l.split('\t'));
}

async function parseCsv(text: string): Promise<string[][]> {
  const { default: Papa } = await import('papaparse');
  const out = Papa.parse<string[]>(text.replace(/^\uFEFF/, ''), { skipEmptyLines: 'greedy' });
  if (out.errors.length > 0 && out.data.length === 0) throw new ParseError(`Couldn’t read the CSV: ${out.errors[0]?.message ?? 'unknown error'}`);
  return out.data;
}

async function parseXlsx(buffer: ArrayBuffer): Promise<string[][]> {
  const XLSX = await import('xlsx');
  const book = XLSX.read(buffer, { type: 'array' });
  const first = book.SheetNames[0];
  const sheet = first ? book.Sheets[first] : undefined;
  if (!sheet) throw new ParseError('The workbook has no sheets.');
  const rows = XLSX.utils.sheet_to_json<unknown[]>(sheet, { header: 1, raw: false, defval: '', blankrows: false });
  return rows.map((r) => r.map((c) => (c == null ? '' : String(c))));
}

/** Reads a File (upload) or a named text (sample data) into a table. */
export async function parseTable(source: File | { name: string; text: string }): Promise<ParsedTable> {
  const isFile = source instanceof File;
  const size = isFile ? source.size : new Blob([source.text]).size;
  if (size > MAX_BYTES) {
    throw new ParseError(`“${source.name}” is ${(size / 1024 / 1024).toFixed(1)} MB. The browser demo reads files up to ${MAX_BYTES / 1024 / 1024} MB.`);
  }
  const format = formatOf(source.name);
  let table: string[][];
  if (format === 'XLSX') {
    if (!isFile) throw new ParseError('Excel data must come from a file.');
    table = await parseXlsx(await source.arrayBuffer());
  } else {
    const text = isFile ? await source.text() : source.text;
    table = format === 'TSV' ? parseTsv(text) : await parseCsv(text);
    // A .txt or .csv that is really tab-separated: re-read it as TSV.
    if (format === 'CSV' && table[0]?.length === 1 && text.includes('\t')) table = parseTsv(text);
  }
  const [header, ...rows] = table;
  if (!header || header.every((h) => h.trim() === '')) throw new ParseError(`“${source.name}” is empty.`);
  return { fileName: source.name, size, format, columns: header.map((h) => h.trim()), rows };
}
