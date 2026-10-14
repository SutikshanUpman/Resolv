#!/bin/bash
# usage: run_bg.sh <logname> <python args...>   (runs detached, logs to amlc/logs/<logname>.log)
cd /teamspace/studios/this_studio/amlc/code
name=$1; shift
nohup /home/zeus/miniconda3/envs/cloudspace/bin/python -u "$@" > ../logs/$name.log 2>&1 &
echo "started $name pid $!"
