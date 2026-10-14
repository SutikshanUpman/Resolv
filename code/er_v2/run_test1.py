"""Test 1 driver: stage-2 v3 -> incremental features f3 -> level-1 -> CE (new pairs only) + stacker -> summary.

    ER_STAGE2=s3 ER_FEAT=f3 ER_OUT_DIR=<dir> python -m er_v2.run_test1
Every stage caches its outputs under new version tags; v2 artifacts are never overwritten.
"""
from .config import STAGE2_VERSION, FEAT_VERSION, log


def main():
    assert STAGE2_VERSION >= 's3' and FEAT_VERSION >= 'f3', 'set ER_STAGE2=s3 ER_FEAT=f3'
    from . import stage2_v3, features, level1, stack, summary
    log('1/5 stage 2 v3'); stage2_v3.run()
    log('2/5 features (incremental)'); features.run()
    log('3/5 level-1'); level1.run()
    log('4/5 CE scoring of new pairs + stacker + decision + submission'); stack.run()
    log('5/5 summary'); summary.run()


if __name__ == '__main__':
    main()
