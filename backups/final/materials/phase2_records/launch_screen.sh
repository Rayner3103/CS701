#!/bin/bash
# Local screen (7,512 train, cos160, stop e150, ckpt every 10) of the BP7 recipe with extra flags; optional smoke first.
# Usage: launch_screen.sh <run_name> <smoke:0|1> [extra train.py flags...]
set -u
NAME=$1; SMOKE=$2; shift 2
RUN=/mnt/hdd/junkaichen/CS701/cs701_runs/$NAME
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python
SRC=/mnt/hdd/junkaichen/CS701/CS701   # current working tree (has aug, init-weights, per-class sigma, stage weights)
mkdir -p $RUN/code && rsync -a --exclude __pycache__ --exclude '*.pyc' $SRC/train.py $SRC/predict.py $SRC/pp_study.py $SRC/pp_eval.py $SRC/make_submission.py $SRC/score_submission.py $SRC/sarbench $SRC/tools $RUN/code/
cd $RUN/code
export NCCL_P2P_DISABLE=1 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/mnt/hdd/junkaichen/CS701/CS701/.runtime/tmp
echo "$NAME extra flags: $*" > $RUN/flags.txt
COMMON=(--data /mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K --train-split train --seed 0
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 80 --lora-alpha 160
  --cascade-rep 1536 --train-norms --backbone-lr 0.0003 --lr 0.0001 --weight-decay 0.05 --cascade --aligned-roi --rpn-nwd 8
  --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3
  --grad-ckpt --spatial-prior 64 --warmup-epochs 1 --schedule-epochs 160)
if [ "$SMOKE" = "1" ]; then
  echo "smoke start $(date '+%F %T')" >> $RUN/launch.log
  CUDA_VISIBLE_DEVICES=0 nice -n 10 $P train.py "${COMMON[@]}" "$@" --epochs 1 --limit 24 --batch-size 4 --workers 2 --eval-splits val --eval-batch-size 4 --save none --out $RUN --name smoke > $RUN/smoke.log 2>&1
  rc=$?; echo "smoke rc=$rc $(date '+%F %T')" >> $RUN/launch.log
  if [ $rc -ne 0 ]; then echo "train rc=SMOKE_FAILED $(date '+%F %T')" >> $RUN/launch.log; exit 1; fi
fi
echo "train start $(date '+%F %T')" >> $RUN/launch.log
nice -n 10 $P -m torch.distributed.run --standalone --nproc_per_node 2 train.py "${COMMON[@]}" "$@" \
  --epochs 160 --stop-after-epoch 150 --save-every 10 --save-training-state --reproducible-loader \
  --batch-size 16 --workers 2 --eval-splits val --eval-every 10 --eval-batch-size 4 \
  --out $RUN --name schedule160 > $RUN/schedule160.stdout.log 2>&1
echo "train rc=$? $(date '+%F %T')" >> $RUN/launch.log
