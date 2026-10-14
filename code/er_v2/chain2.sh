#!/bin/bash
# CE training starts immediately (needs only train stage-2); features/level-1 on CPU; CE scoring on GPU; then stacker.
cd /teamspace/studios/this_studio/amlc/code
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
L=../logs
$PY -u -m er_v2.stack e5b --ce-train-only > $L/ce_train.log 2>&1 &
CT=$!
while pgrep -f "er_v2.stage2" > /dev/null; do sleep 10; done
[ -f ../output/blocking/test_stage2_s2a.parquet ] || { echo "stage2 test missing" > $L/chain2.err; exit 1; }
$PY -u -m er_v2.features train > $L/feat_train.log 2>&1 &
FT=$!
$PY -u -m er_v2.features test > $L/feat_test.log 2>&1 &
FE=$!
wait $CT || { echo "ce train failed" > $L/chain2.err; exit 1; }
$PY -u -m er_v2.stack e5b --ce-only > $L/ce.log 2>&1 &
CS=$!
wait $FT $FE
$PY -u -m er_v2.level1 > $L/level1.log 2>&1 || { echo "level1 failed" > $L/chain2.err; exit 1; }
wait $CS || { echo "ce score failed" > $L/chain2.err; exit 1; }
$PY -u -m er_v2.stack e5b > $L/stack.log 2>&1 || { echo "stack failed" > $L/chain2.err; exit 1; }
echo done > $L/chain2.done
