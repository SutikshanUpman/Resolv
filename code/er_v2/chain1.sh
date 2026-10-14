#!/bin/bash
# after dense retrieval: stage-2 -> [CE train+score on GPU] || [features train/test -> level-1 on CPU] -> stacker
cd /teamspace/studios/this_studio/amlc/code
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
L=../logs
while pgrep -f "er_v2.dense" > /dev/null; do sleep 10; done
[ -f ../output/blocking/test_dense_d1_k40_3.parquet ] || { echo "dense output missing" > $L/chain1.err; exit 1; }
$PY -u -m er_v2.stage2 > $L/stage2.log 2>&1 || { echo "stage2 failed" > $L/chain1.err; exit 1; }
$PY -u -m er_v2.stack e5b --ce-only > $L/ce.log 2>&1 &
CE=$!
( $PY -u -m er_v2.features train > $L/feat_train.log 2>&1 & $PY -u -m er_v2.features test > $L/feat_test.log 2>&1 & wait ) 
$PY -u -m er_v2.level1 > $L/level1.log 2>&1 || { echo "level1 failed" > $L/chain1.err; }
wait $CE || { echo "ce failed" > $L/chain1.err; exit 1; }
$PY -u -m er_v2.stack e5b > $L/stack.log 2>&1 || { echo "stack failed" > $L/chain1.err; exit 1; }
echo done > $L/chain1.done
