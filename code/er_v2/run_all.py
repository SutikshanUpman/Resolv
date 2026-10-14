"""End-to-end driver: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

Every stage caches its artifacts (versioned parquet / model files), so re-running resumes after the last finished stage.

    ER_DATA_DIR=/path/to/dataset ER_ROOT=/path/to/workdir python -m er_v2.run_all [e5b]
"""
import sys
from .config import log


def main():
    from . import normalize, splits, blocking_sparse, dense, stage2, features, level1, stack
    log('1/8 normalise'); normalize.run()
    log('2/8 ghost split + subsets'); splits.run()
    log('3/8 sparse blocking'); blocking_sparse.run()
    log('4/8 dense retrieval'); dense.run()
    log('5/8 union + stage-2 re-rank + cut'); stage2.run()
    log('6/8 pair features'); features.run()
    log('7/8 level-1 + competition features'); level1.run()
    log('8/8 cross-encoder + stacker + decision + submission'); stack.run()


if __name__ == '__main__':
    main()
