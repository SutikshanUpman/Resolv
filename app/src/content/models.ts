import { docText, fact, type DocText, type Fact } from './fact';

export type ModelId = 'e5Small' | 'e5Base' | 'qwen' | 'lightgbm';

export interface Model {
  readonly id: ModelId;
  readonly name: DocText;
  readonly role: string;
  readonly licence: DocText;
  /** Parameter count in millions; tree ensembles have none. */
  readonly params?: Fact;
}

// §4.1 architecture table
export const models: Readonly<Record<ModelId, Model>> = {
  e5Small: {
    id: 'e5Small',
    name: docText('intfloat/multilingual-e5-small', '4.1'),
    role: 'Dense bi-encoder for retrieval',
    licence: docText('MIT', '4.1'),
    params: fact(118, '118M', '4.1'),
  },
  e5Base: {
    id: 'e5Base',
    name: docText('intfloat/multilingual-e5-base', '4.1'),
    role: 'Cross-encoder',
    licence: docText('MIT', '4.1'),
    params: fact(278, '278M', '4.1'),
  },
  qwen: {
    id: 'qwen',
    name: docText('Qwen/Qwen3-Reranker-0.6B', '4.1'),
    role: 'LLM reranker on uncertain pairs',
    licence: docText('Apache-2.0', '4.1'),
    params: fact(596, '596M', '4.1'),
  },
  lightgbm: {
    id: 'lightgbm',
    name: docText('LightGBM', '4.1'),
    role: 'Re-ranker, level-1, stacker and collective models',
    licence: docText('MIT', '4.1'),
  },
};
