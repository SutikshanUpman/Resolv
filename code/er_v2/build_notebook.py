"""Builds MLChallenge_v2.ipynb: each pipeline stage as a cell (cached -> re-running is fast) followed by its report."""
import json, sys
from pathlib import Path


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')}


C = []
C.append(md(r"""
# Amazon ML Challenge 2026: Business Entity Resolution (v2)

For every Source-1 (S1) business, find all matching Source-2/3 records. The metric is **per-S1 F0.5, macro-averaged over all S1**; singletons score 1 only for an empty prediction.

**Why v2.** v1 reached 0.963 on a held-out train split but ~0.94 on the leaderboard. The EDA below shows the main reason:
* test has **5.75 S2/S3 records per S1 against 4.68 in train**, while its true matches per S1 are unchanged (≈3.45);
* so about **41 % of test records match no S1, against 26 % in train**;
* a model tuned on train over-predicts on test, merging orphan records into a same-name S1 at another address.

v2 trains and tunes everything on a *test-like* train distribution, and rebuilds blocking and matching around it:

```
normalise → ghost-S1 split → sparse keys (11 families) ∪ dense bi-encoder (e5-small) → LightGBM re-rank + adaptive cut = candidate_pairs.tsv
          → 94 pair features → level-1 LightGBM (full graph, cross-fitted) + competition features
          → fine-tuned cross-encoder (multilingual-e5-base) → LightGBM stacker → global exclusive assignment + OOF threshold = matching_results.tsv
```

Every stage caches versioned artifacts, so running this notebook top to bottom resumes after the last finished stage. The source is in `er_v2/`. Execution: Lightning AI A100-40GB, 30 vCPU, 216 GB RAM.
"""))
C.append(code(r"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.abspath('code') if os.path.isdir('code/er_v2') else os.path.abspath('.'))
import polars as pl, pandas as pd
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(80)
from er_v2.config import OUT, FINAL_OUT, DATA, log
CE_TAGS = os.environ.get('ER_CE_TAGS', 'e5b').split(',')
print('cross-encoders:', CE_TAGS)
print('data:', DATA, '| artifacts:', OUT, '| submission:', FINAL_OUT)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
"""))
C.append(md(r"""
## 1. Data and the train/test shift (EDA)
Group sizes, singleton rate and the per-country record/S1 ratio. The test distractor share is estimated from exact normalised-name hit rates:
* a record whose normalised name equals some S1 name is far more likely to be matched (60 %) than a distractor (31 %, India);
* solving the mixture on test India gives 3.45 matched records per S1, the same as train, so the extra test records are distractors.
"""))
C.append(code(r"""
from er_v2.normalize import read_tsv
from er_v2.config import raw_path
cnt = []
for split in ['train', 'test']:
    for s in '123':
        d = pl.scan_csv(raw_path(f'{split}_source{s}'), separator='\t', quote_char=None, infer_schema=False).group_by('country').len().collect()
        cnt += [{'split': split, 'source': f'S{s}', 'country': c, 'rows': n} for c, n in d.iter_rows()]
cnt = pl.DataFrame(cnt)
piv = cnt.pivot(on='source', index=['split', 'country'], values='rows').with_columns(((pl.col('S2') + pl.col('S3')) / pl.col('S1')).round(3).alias('records per S1'))
piv.sort('split', 'country')
"""))
C.append(md("## 2. Normalisation (v2) and the ghost-S1 split"))
C.append(code(r"""
from er_v2 import normalize, splits
normalize.run(); splits.run()
print(json.dumps(json.load(open(OUT / 'ghost_info.json')), indent=1))
from er_v2.data import nt
nt('test', '1', ['country', 'name', 'name_core', 'name_sk', 'addr', 'addr_norm', 'loc', 'addr_nums']).sample(8, seed=1)
"""))
C.append(md("## 3. Candidate generation: sparse keys ∪ dense retrieval → learned cut"))
C.append(code(r"""
from er_v2 import blocking_sparse, dense, stage2
blocking_sparse.run(); dense.run(); stage2.run()
print('rr_tune (stage-2 tuning):'); display(pl.read_csv(OUT / 'blocking_report_rr_tune.csv'))
print('val (never used by blocking):'); display(pl.read_csv(OUT / 'blocking_report_val.csv'))
cfg = json.load(open(OUT / 'models' / 'stage2_cfg_s2a.json')); print({k: v for k, v in cfg.items() if k != 'RR_FEATS'})
"""))
C.append(md("## 4. Pair features, level-1 model and competition features"))
C.append(code(r"""
from er_v2 import features, level1
features.run(); level1.run()
print(json.load(open(OUT / 'models' / 'level1_f2' / 'info.json')))
"""))
C.append(md(r"""
## 5. Cross-encoder, stacker, decision rule, submission
* The cross-encoders are fine-tuned on the `ce_train` candidates.
* They score the evaluation set E = stack_train ∪ val ∪ closure (every S1 competing for their records) and all of test.
* The LightGBM stacker is fitted with 5-fold OOF on stack_train. The threshold under global exclusive assignment is chosen on OOF.
* val is used only for the report.
"""))
C.append(code(r"""
from er_v2 import stack
stack.CE_TAGS = CE_TAGS
stack.run()
"""))
C.append(md("## 6. Decision-rule study (chosen on stack_train OOF; val for reporting only)"))
C.append(code(r"""
from er_v2 import decide
decide.TAG = '_'.join(CE_TAGS)
study = decide.run(write=False)
pd.DataFrame({k: {'oof': v['oof'], 'val': v['val']['all'] if isinstance(v['val'], dict) else v['val']} for k, v in study.items()}).T
"""))
C.append(md("## 7. Results summary"))
C.append(code(r"""
hist = sorted((OUT / 'submission_history').iterdir())[-1]
info = json.load(open(hist / 'run_info.json'))
rows = [
    ('v1 (submitted, LB ~0.94)', 'e5-small CE + LightGBM', 'raw train (26 % distractors)', 0.96310, 0.97149, 4.79),
    ('v2: level-1 LightGBM, features only', 'LightGBM', 'ghost train (41 % distractors)', info['level1_only_val'], info['val_oracle']['oracle F0.5'], info['val_oracle']['cand / S1']),
    ('v2: cross-encoder only', 'multilingual-e5-base CE', 'ghost train', list(info['ce_only_val'].values())[0], info['val_oracle']['oracle F0.5'], info['val_oracle']['cand / S1']),
    ('v2: FINAL stack', 'LightGBM stack + CE(s)', 'ghost train', info['val']['all'], info['val_oracle']['oracle F0.5'], info['val_oracle']['cand / S1']),
]
display(pd.DataFrame(rows, columns=['experiment', 'model', 'validation distribution', 'val macro F0.5', 'oracle F0.5', 'cand / S1']))
print('val by country:', {k: round(v, 5) for k, v in info['val'].items()})
display(pd.DataFrame(info['test_by_country']))
"""))
C.append(md("## 8. Official validator"))
C.append(code(r"""
r = subprocess.run([sys.executable, 'code/validate_submission.py', '--matching', str(FINAL_OUT / 'matching_results.tsv'),
                    '--candidate', str(FINAL_OUT / 'candidate_pairs.tsv'), '--test-dir', str(DATA), '--check-ids'], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
"""))
nb = {'cells': C, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = Path(sys.argv[1] if len(sys.argv) > 1 else 'MLChallenge_v2.ipynb')
out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('wrote', out)
