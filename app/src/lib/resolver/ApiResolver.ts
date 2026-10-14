import type { Resolver, ResolverEvent, ResolverInput, ResolverRun } from './types';

/**
 * Placeholder for a real backend. A server would accept the three sources (for example POST {VITE_API_URL}/resolve)
 * and stream ResolverEvent objects back (server-sent events or NDJSON), ending with { type: 'done', result }.
 * Until one exists, it reports that it is not configured.
 */
export class ApiResolver implements Resolver {
  readonly kind = 'api';
  readonly liveInference = true;
  private readonly baseUrl: string | undefined;

  constructor(baseUrl: string | undefined) {
    this.baseUrl = baseUrl;
  }

  resolve(input: ResolverInput): ResolverRun {
    const baseUrl = this.baseUrl;
    async function* events(): AsyncGenerator<ResolverEvent> {
      await Promise.resolve();
      yield {
        type: 'error',
        code: 'NOT_CONFIGURED',
        message: baseUrl
          ? `The API resolver at ${baseUrl} is not implemented yet (${input.s1.length} businesses were ready to send).`
          : 'No backend is configured. Set VITE_API_URL, or use VITE_RESOLVER=demo.',
      };
    }
    return { events: events(), skip: () => undefined, cancel: () => undefined };
  }
}
