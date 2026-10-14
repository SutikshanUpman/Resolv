import { funnel } from '../../content/blocking';
import { qwenBand, type StageId } from '../../content/pipeline';
import { splitLabels } from '../../content/labels';
import { formatFact } from '../format';
import type { CounterKey } from '../resolver/types';

export type StageStatus = 'waiting' | 'running' | 'done';

export const counterLabels: Record<CounterKey, string> = {
  businesses: 'businesses loaded',
  records: 'records loaded',
  candidatesAfterCut: 'candidates kept after the cut',
  matchesAssigned: 'matches assigned',
  unmatched: 'businesses left unmatched',
};

/** Stages without a per-run count show the doc's figure instead, labelled with its split. */
export const stageNotes: Partial<Record<StageId, string>> = {
  union: `About ${formatFact(funnel.union, { digits: 1 })} candidates per business (${splitLabels.validation.short.toLowerCase()} average)`,
  llmRerank: `Scores only uncertain pairs: ${qwenBand.text}`,
};

export const stageCounters: Partial<Record<StageId, CounterKey[]>> = {
  normalise: ['businesses', 'records'],
  rerankCut: ['candidatesAfterCut'],
  assign: ['matchesAssigned', 'unmatched'],
};
