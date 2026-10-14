"""Paths, seeds and cache versions shared by every stage of the v2 pipeline.

Every expensive artifact is written as parquet under OUT with a version tag, so a re-run resumes from the last
finished stage and a changed stage never silently reuses a stale cache (bump its version constant instead).
"""
import os, time, json
from pathlib import Path

ROOT = Path(os.environ.get('ER_ROOT', '/teamspace/studios/this_studio/amlc'))
DATA = Path(os.environ.get('ER_DATA_DIR', ROOT / 'data'))          # the 7 raw TSVs (flat)
OUT = Path(os.environ.get('ER_WORK_DIR', ROOT / 'output'))         # all cached artifacts
FINAL_OUT = Path(os.environ.get('ER_OUT_DIR', ROOT.parent / 'output'))  # matching_results.tsv + candidate_pairs.tsv
for d in [OUT, FINAL_OUT]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
GHOST_FRAC = 0.20        # share of train S1 entities removed so train has the test-like distractor rate (~41 %)
TEAM_NAME = os.environ.get('ER_TEAM', 'team')

NORM_VERSION = 'v2'
BLOCK_VERSION = 'b2'
DENSE_VERSION = 'd1'
STAGE2_VERSION = os.environ.get('ER_STAGE2', 's2a')   # v3 test runs set ER_STAGE2=s3
FEAT_VERSION = os.environ.get('ER_FEAT', 'f2')          # v3 test runs set ER_FEAT=f3
L1_TAG = os.environ.get('ER_L1', '')                    # Test 2: 'b' = level-1 on all rows, finer settings
STACK_TAG = os.environ.get('ER_STACK', '')              # Test 2: 'r' = + CE record-side competition features

RAW_FILES = {'train_source1': 2206821, 'train_source2': 5034616, 'train_source3': 5285603, 'train_ground_truth': 2206821,
             'test_source1': 1732544, 'test_source2': 4887273, 'test_source3': 5082316}

T0 = time.time()


def raw_path(name):
    """Path of a raw TSV, e.g. raw_path('train_source1'); accepts a flat folder or train/ + test/ sub-folders."""
    p = DATA / f'{name}.tsv'
    return p if p.exists() else DATA / name.split('_')[0] / f'{name}.tsv'


def log(*a):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *a, flush=True)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))
