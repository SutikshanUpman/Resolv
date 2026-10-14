import { useSyncExternalStore } from 'react';
import type { ParsedTable } from '../parse/parseFile';
import type { Validation } from '../parse/validate';
import type { ResolveResult, SourceId } from '../resolver/types';
import { fetchSampleRecords } from './sampleFiles';
import { completeResult, loadStoredResult, saveResult } from './persist';

export type SlotState =
  | { status: 'empty' }
  | { status: 'parsing'; fileName: string }
  | { status: 'ready'; origin: 'sample' | 'upload'; table: ParsedTable; validation: Validation }
  | { status: 'error'; fileName: string; message: string };

export interface WorkspaceState {
  slots: Record<SourceId, SlotState>;
  result: ResolveResult | null;
  /** A result saved in this tab is being rebuilt after a reload. */
  restoring: boolean;
}

const stored = loadStoredResult();

let state: WorkspaceState = {
  slots: { S1: { status: 'empty' }, S2: { status: 'empty' }, S3: { status: 'empty' } },
  result: stored.kind === 'complete' ? stored.result : null,
  restoring: stored.kind === 'needsRecords',
};
const listeners = new Set<() => void>();

/** Workspace state shared by the upload, run and results pages. The last result is also kept in sessionStorage. */
export const workspace = {
  get: () => state,
  subscribe(listener: () => void) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  update(fn: (s: WorkspaceState) => WorkspaceState) {
    state = fn(state);
    for (const l of listeners) l();
  },
  setSlot(source: SourceId, slot: SlotState) {
    workspace.update((s) => ({ ...s, slots: { ...s.slots, [source]: slot }, result: null, restoring: false }));
    saveResult(null);
  },
  setResult(result: ResolveResult | null) {
    workspace.update((s) => ({ ...s, result, restoring: false }));
    saveResult(result);
  },
};

// The saved result held IDs only (over the storage quota): rebuild its records from the sample files.
if (stored.kind === 'needsRecords') {
  fetchSampleRecords().then(
    (records) => state.restoring && workspace.update((s) => ({ ...s, result: completeResult(stored.stored, records), restoring: false })),
    () => state.restoring && workspace.update((s) => ({ ...s, restoring: false })),
  );
}

export function useWorkspace(): WorkspaceState {
  return useSyncExternalStore(workspace.subscribe, workspace.get);
}

/** All three sources parsed and valid. */
export function isReady(s: WorkspaceState): boolean {
  return Object.values(s.slots).every((slot) => slot.status === 'ready' && slot.validation.errors.length === 0);
}

/** Any source came from the visitor rather than the sample data. */
export function hasUpload(s: WorkspaceState): boolean {
  return Object.values(s.slots).some((slot) => slot.status === 'ready' && slot.origin === 'upload');
}
