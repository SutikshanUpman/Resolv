import { ApiResolver } from './ApiResolver';
import { DemoResolver } from './DemoResolver';
import type { Resolver } from './types';

/** The one resolver the UI talks to. VITE_RESOLVER=api switches to the backend stub; the default is the demo. */
export const resolver: Resolver =
  import.meta.env.VITE_RESOLVER === 'api' ? new ApiResolver(import.meta.env.VITE_API_URL) : new DemoResolver();

export type * from './types';
