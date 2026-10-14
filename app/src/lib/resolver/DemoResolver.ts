import { pipeline, type StageId } from '../../content/pipeline';
import { parseIdListTsv, summarise } from './summary';
import type { BusinessRecord, CounterKey, Resolver, ResolverEvent, ResolverInput, ResolverRun, ResolveResult } from './types';

/** How long each stage plays, in ms: about seven seconds in all. */
const STAGE_MS: Record<StageId, number> = {
  normalise: 700,
  retrieve: 900,
  union: 500,
  rerankCut: 900,
  pairFeatures: 600,
  crossEncoder: 700,
  stacker: 500,
  llmRerank: 800,
  collective: 500,
  assign: 900,
};

async function fetchText(path: string): Promise<string> {
  const r = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.text();
}

/**
 * Replays our real submission for the sample data. It never runs a model, so it refuses a visitor's own
 * files: the UI must not suggest it ran inference on them.
 */
export class DemoResolver implements Resolver {
  readonly kind = 'demo';
  readonly liveInference = false;

  resolve(input: ResolverInput): ResolverRun {
    let skipped = false;
    let cancelled = false;
    let wake: (() => void) | null = null;
    const wait = (ms: number) =>
      skipped || cancelled
        ? Promise.resolve()
        : new Promise<void>((resolve) => {
            const t = window.setTimeout(resolve, ms);
            wake = () => {
              window.clearTimeout(t);
              resolve();
            };
          });

    async function* events(): AsyncGenerator<ResolverEvent> {
      if (input.origin === 'upload') {
        yield {
          type: 'error',
          code: 'LIVE_INFERENCE_UNAVAILABLE',
          message: 'Live inference isn’t connected in this demo. Your files were checked, but no model ran on them.',
        };
        return;
      }

      let result: ResolveResult;
      try {
        const [candText, matchText] = await Promise.all([
          fetchText('demo/candidate_pairs.tsv'),
          fetchText('demo/matching_results.tsv'),
        ]);
        const candidates = parseIdListTsv(candText);
        const matches = parseIdListTsv(matchText);
        const records = new Map<string, BusinessRecord>();
        for (const r of [...input.s1, ...input.s2, ...input.s3]) records.set(r.id, r);
        const missing = candidates.find((c) => !records.has(c.s1Id) || c.ids.some((id) => !records.has(id)));
        if (missing) {
          yield {
            type: 'error',
            code: 'INPUT_MISMATCH',
            message: `The sample files don’t match the precomputed results (first mismatch: ${missing.s1Id}). Reload the sample data.`,
          };
          return;
        }
        result = { mode: 'demo', candidates, matches, records, summary: summarise(matches) };
      } catch (err) {
        yield { type: 'error', code: 'LOAD_FAILED', message: err instanceof Error ? err.message : String(err) };
        return;
      }

      const totals: Partial<Record<StageId, [CounterKey, number][]>> = {
        normalise: [
          ['businesses', input.s1.length],
          ['records', input.s2.length + input.s3.length],
        ],
        rerankCut: [['candidatesAfterCut', result.candidates.reduce((n, r) => n + r.ids.length, 0)]],
        assign: [
          ['matchesAssigned', result.summary.matches],
          ['unmatched', result.matches.filter((r) => r.ids.length === 0).length],
        ],
      };

      for (const stage of pipeline) {
        if (cancelled) return;
        yield { type: 'stage-start', stage: stage.id };
        const counters = totals[stage.id] ?? [];
        const ticks = 8;
        for (let t = 1; t <= ticks; t++) {
          await wait(STAGE_MS[stage.id] / ticks);
          if (cancelled) return;
          for (const [counter, total] of counters) {
            const value = skipped ? total : Math.round((total * t) / ticks);
            yield { type: 'progress', stage: stage.id, counter, value, total };
          }
          if (skipped) break;
        }
        for (const [counter, total] of counters) yield { type: 'progress', stage: stage.id, counter, value: total, total };
        yield { type: 'stage-end', stage: stage.id };
      }
      yield { type: 'done', result };
    }

    return {
      events: events(),
      skip: () => {
        skipped = true;
        wake?.();
      },
      cancel: () => {
        cancelled = true;
        wake?.();
      },
    };
  }
}
