#!/bin/bash
# level-1 (restarted) then, once CE scoring is done, the stacker
cd /teamspace/studios/this_studio/amlc/code
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
L=../logs
$PY -u -m er_v2.level1 > $L/level1.log 2>&1 || { echo "level1 failed" > $L/chain3.err; exit 1; }
while pgrep -f "er_v2.stack e5b --ce-only" > /dev/null; do sleep 10; done
$PY -u -m er_v2.stack e5b > $L/stack.log 2>&1 || { echo "stack failed" > $L/chain3.err; exit 1; }
echo done > $L/chain3.done
