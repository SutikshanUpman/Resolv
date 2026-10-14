"""Test 2 driver (on Test 1's candidates and features): level-1 on all rows -> stacker + CE record-side features ->
collective second pass + rank-aware decision -> summary.

    ER_STAGE2=s3 ER_FEAT=f3 ER_L1=b ER_STACK=r ER_OUT_DIR=<dir> python -m er_v2.run_test2
"""
from .config import L1_TAG, STACK_TAG, FEAT_VERSION, STAGE2_VERSION, log


def main():
    assert L1_TAG and STACK_TAG, 'set ER_L1=b ER_STACK=r'
    from . import level1, stack, collective, summary
    log('1/4 level-1 (all rows)'); level1.run()
    log('2/4 stacker + CE record-side features'); stack.run()
    s_stack = summary.run()
    log('3/4 collective second pass'); best, study = collective.run()
    log('4/4 summary'); summary.run(ev_name=f'eval_p_{FEAT_VERSION}{L1_TAG}{STACK_TAG}_coll_e5b.parquet' if best.startswith('collective') else None)


if __name__ == '__main__':
    main()
