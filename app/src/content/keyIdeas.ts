import { docText, fact } from './fact';

// §1 three key ideas
export const keyIdeas = [
  {
    id: 'ghost',
    title: docText('Train on a test-like distribution.', '1'),
    text: docText(
      'The test set contains far more unmatched records than train. We remove 20 % of train S1 ("ghost S1") so that their records become unmatched noise, and we fit every model and threshold on that distribution.',
      '1',
    ),
    figure: fact(20, '20 %', '1', { context: 'We remove 20 % of train S1' }),
  },
  {
    id: 'exclusive',
    title: docText('One record, one business.', '1'),
    text: docText(
      'A record can match at most one S1. At every stage, each candidate pair therefore also sees how strongly other S1 claim the same record, and the final assignment gives each record to at most one S1.',
      '1',
    ),
  },
  {
    id: 'llm',
    title: docText('A multilingual LLM reranker where the other models are unsure.', '1'),
    text: docText('fine-tuned on our training pairs, scores only uncertain pairs.', '1'),
    before: fact(0.9834, '0.9834', '1', { context: '0.9834 → 0.985619', split: 'leaderboard' }),
    after: fact(0.985619, '0.985619', '1', { context: '0.9834 → 0.985619', split: 'leaderboard' }),
  },
] as const;
