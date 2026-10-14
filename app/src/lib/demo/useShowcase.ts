import { useEffect, useState } from 'react';
import type { ShowcaseExample, ShowcaseFile } from './types';

let cache: Promise<ShowcaseExample[]> | null = null;

function loadShowcase(): Promise<ShowcaseExample[]> {
  cache ??= fetch(`${import.meta.env.BASE_URL}demo/showcase.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`showcase.json: HTTP ${r.status}`);
      return r.json() as Promise<ShowcaseFile>;
    })
    .then((f) => f.examples)
    .catch((err: unknown) => {
      cache = null;
      throw err;
    });
  return cache;
}

export type ShowcaseState =
  | { status: 'loading' }
  | { status: 'ready'; examples: ShowcaseExample[] }
  | { status: 'error'; message: string };

/** The landing-page examples, fetched once and shared by every component that asks. */
export function useShowcase(): ShowcaseState {
  const [state, setState] = useState<ShowcaseState>({ status: 'loading' });
  useEffect(() => {
    let live = true;
    loadShowcase().then(
      (examples) => live && setState({ status: 'ready', examples }),
      (err: unknown) => live && setState({ status: 'error', message: err instanceof Error ? err.message : String(err) }),
    );
    return () => {
      live = false;
    };
  }, []);
  return state;
}
