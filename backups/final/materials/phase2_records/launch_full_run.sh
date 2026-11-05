#!/bin/bash
# Full-data (9,392) BP7 cos200 run with optional extra flags; mirrors the 56.49 trajectory's recipe.
# Usage: launch_full_run.sh <run_name> [extra train.py flags...]   e.g. launch_full_run.sh FULL_AUG1 --speckle 0.2 --gamma-jitter 0.29 --photo-p 0.5
set -u
NAME=$1; shift
RUN=/mnt/hdd/junkaichen/CS701/cs701_runs/$NAME
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python
SRC=/mnt/hdd/junkaichen/CS701/cs701_runs/AUG1_speckle_gamma_local7512_cos160_s0_20261101/code   # frozen snapshot with the aug feature
mkdir -p $RUN && rsync -a $SRC/ $RUN/code/
cd $RUN/code
export NCCL_P2P_DISABLE=1 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/mnt/hdd/junkaichen/CS701/CS701/.runtime/tmp
echo "$NAME extra flags: $*" > $RUN/flags.txt
echo "train start $(date '+%F %T')" >> $RUN/launch.log
nice -n 10 taskset -c 0-7 $P -m torch.distributed.run --standalone --nproc_per_node 2 train.py \
  --data /mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K --train-split train+val+test --eval-splits '' --seed 0 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 80 --lora-alpha 160 \
  --cascade-rep 1536 --train-norms --backbone-lr 0.0003 --lr 0.0001 --weight-decay 0.05 --cascade --aligned-roi --rpn-nwd 8 \
  --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3 \
  --grad-ckpt --spatial-prior 64 --warmup-epochs 1 --epochs 200 --schedule-epochs 200 --stop-after-epoch 190 \
  --save-every 5 --save-training-state --reproducible-loader --compile --batch-size 16 --workers 2 --eval-batch-size 4 \
  "$@" --out $RUN --name full_train > $RUN/full_train.stdout.log 2>&1
echo "train rc=$? $(date '+%F %T')" >> $RUN/launch.log
