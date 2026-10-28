#!/bin/bash
# Retrain SC2: F2's full-data recipe (all 9,392 labelled images, cosine, 2-GPU DDP, global batch 16) scaled to the
# parameter budget (LoRA r80 / alpha 160, Cascade head width 1536; 124.9M total / 39.7M trainable), 150 epochs with a
# checkpoint every 10. The backed-up model is ckpt_e130.pt (epoch 130 of this 150-epoch cosine schedule).
# --grad-ckpt only saves memory (peak 10.6 GB per GPU); drop it on free 48 GB GPUs. 6.4 h on 2 x RTX 6000 Ada.
# DATA = the converted data from prepare_course_data.py (default: the setup.md layout next to the repo).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$HERE/../../../dataset/SARFact-Course-20K}
TORCHRUN=${TORCHRUN:-torchrun}
cd "$HERE/code"
NCCL_P2P_DISABLE=1 $TORCHRUN --standalone --nproc_per_node 2 train.py \
  --data "$DATA" --out "$HERE/runs" --name SC2_full150 \
  --train-split train+val+test --eval-splits '' --save-every 10 --epochs 150 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 80 --lora-alpha 160 --cascade-rep 1536 \
  --train-norms --backbone-lr 3e-4 --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3 --grad-ckpt
