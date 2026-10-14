#!/bin/bash
# Final pass (see plan): Stage A = Test-3 Qwen model on the wide band (0.02-0.98), collective with global threshold;
# Stage B = Test-3 Qwen model continued on generically-noised training pairs, scored on the wide band, collective with both.
# GPU steps run in sequence (A-score -> B-train -> B-score); the Stage A collective (CPU) runs while B trains.
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
C=/teamspace/studios/this_studio/amlc/code; L=/teamspace/studios/this_studio/amlc/logs; S=/teamspace/studios/this_studio
cd $C
export ER_STAGE2=s3 ER_FEAT=f3 ER_L1=b ER_STACK=r
echo "[final] start $(date)"

# ---- on an 80 GB GPU (H100) Stage B training starts right away, in parallel with Stage A scoring
if [ "${PARALLEL:-0}" = "1" ]; then
  OMP_NUM_THREADS=3 nohup nice -n 10 $PY -u -m er_v2.qwen_ce train_aug > $L/final_B_train.log 2>&1 &
  BPID=$!; echo "[final] B training started in parallel (pid $BPID)"
fi

# ---- Stage A: score only the pairs of the wide band that the Test-3 run did not score (same model, reuse old scores)
ER_QW_BAND=0.02,0.98 ER_QW_MODEL=qwen_rr06 ER_QW_OUT=_wide ER_QW_REUSE= OMP_NUM_THREADS=3 \
  $PY -u -m er_v2.qwen_ce score > $L/final_A_score.log 2>&1 || { echo "[final] A scoring FAILED"; exit 1; }
echo "[final] A scoring done $(date)"

# ---- Stage B training on the GPU, Stage A collective on the CPU at the same time
if [ "${PARALLEL:-0}" != "1" ]; then
  OMP_NUM_THREADS=3 nohup nice -n 10 $PY -u -m er_v2.qwen_ce train_aug > $L/final_B_train.log 2>&1 &
  BPID=$!
fi
mkdir -p $S/output_finalA
ER_QW=1 ER_QW_FILES=_wide:qw ER_QW_TAG=_qwA ER_RULE=global ER_OUT_DIR=$S/output_finalA \
  $PY -u -m er_v2.collective > $L/final_A_collective.log 2>&1 && echo "[final] A collective done $(date)" || echo "[final] A collective FAILED"

wait $BPID || { echo "[final] B training FAILED"; exit 1; }
echo "[final] B training done $(date)"

# ---- Stage B scoring (new model, wide band, no reuse) and collective with both Qwen scores
ER_QW_BAND=0.02,0.98 ER_QW_MODEL=qwen_rr06_aug ER_QW_OUT=_aug OMP_NUM_THREADS=3 \
  $PY -u -m er_v2.qwen_ce score > $L/final_B_score.log 2>&1 || { echo "[final] B scoring FAILED"; exit 1; }
echo "[final] B scoring done $(date)"
mkdir -p $S/output_finalB
ER_QW=1 ER_QW_FILES=_wide:qw,_aug:qw4 ER_QW_TAG=_qwB ER_RULE=global ER_OUT_DIR=$S/output_finalB \
  $PY -u -m er_v2.collective > $L/final_B_collective.log 2>&1 && echo "[final] B collective done $(date)" || echo "[final] B collective FAILED"
echo "[final] end $(date)"
