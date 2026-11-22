#!/bin/bash
# Retrain F2_full: the exact command of the 2026-10-21 run (cs701_runs/queue/done/a3a1_F2_full_ddp.sh) with
# portable paths. 2 GPUs (DDP, 8 images per GPU), about 3.6 h on 2 x RTX 6000 Ada, peak 38 GB per GPU.
# DATA = the converted data from prepare_course_data.py (default: the setup.md layout next to the repo).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$HERE/../../../dataset/SARFact-Course-20K}
PY=${PY:-python}
TORCHRUN=${TORCHRUN:-torchrun}
cd "$HERE/code"
NCCL_P2P_DISABLE=1 $TORCHRUN --standalone --nproc_per_node 2 train.py \
  --data "$DATA" --out "$HERE/runs" --name F2_full \
  --train-split train+val+test --eval-splits '' --save-every 10 --epochs 100 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 32 \
  --lora-alpha 64 --train-norms --backbone-lr 3e-4 --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3
