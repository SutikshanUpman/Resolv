import { ParseError, parseTable } from '../parse/parseFile';
import { validateSource } from '../parse/validate';
import type { SourceId } from '../resolver/types';
import { SOURCES, fetchSampleText } from './sampleFiles';
import { workspace } from './store';

export { SOURCES };

async function load(source: SourceId, input: File | { name: string; text: string }, origin: 'sample' | 'upload') {
  workspace.setSlot(source, { status: 'parsing', fileName: input.name });
  try {
    const table = await parseTable(input);
    workspace.setSlot(source, { status: 'ready', origin, table, validation: validateSource(table, source) });
  } catch (err) {
    const message = err instanceof ParseError || err instanceof Error ? err.message : String(err);
    workspace.setSlot(source, { status: 'error', fileName: input.name, message });
  }
}

export function loadUpload(source: SourceId, file: File): Promise<void> {
  return load(source, file, 'upload');
}

/** Fetches the three sample files and runs them through the same parser and checks as an upload. */
export async function loadSample(): Promise<void> {
  await Promise.all(
    SOURCES.map(async (s) => {
      workspace.setSlot(s.id, { status: 'parsing', fileName: s.sample });
      try {
        await load(s.id, await fetchSampleText(s.sample), 'sample');
      } catch (err) {
        workspace.setSlot(s.id, {
          status: 'error',
          fileName: s.sample,
          message: `The sample file didn’t load (${err instanceof Error ? err.message : String(err)}).`,
        });
      }
    }),
  );
}

export function clearSource(source: SourceId): void {
  workspace.setSlot(source, { status: 'empty' });
}
