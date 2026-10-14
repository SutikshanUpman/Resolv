import { parseTable } from '../parse/parseFile';
import { validateSource } from '../parse/validate';
import type { BusinessRecord, SourceId } from '../resolver/types';

export const SOURCES: readonly { id: SourceId; label: string; role: string; sample: string }[] = [
  { id: 'S1', label: 'Source 1', role: 'Reference businesses', sample: 'sample_source1.tsv' },
  { id: 'S2', label: 'Source 2', role: 'Records to match', sample: 'sample_source2.tsv' },
  { id: 'S3', label: 'Source 3', role: 'Records to match', sample: 'sample_source3.tsv' },
];

export async function fetchSampleText(fileName: string): Promise<{ name: string; text: string }> {
  const r = await fetch(`${import.meta.env.BASE_URL}demo/${fileName}`);
  if (!r.ok) throw new Error(`${fileName}: HTTP ${r.status}`);
  return { name: fileName, text: await r.text() };
}

/** Every record of the three sample files, through the same parser and checks as an upload. */
export async function fetchSampleRecords(): Promise<BusinessRecord[]> {
  const parts = await Promise.all(SOURCES.map(async (s) => validateSource(await parseTable(await fetchSampleText(s.sample)), s.id).records));
  return parts.flat();
}
