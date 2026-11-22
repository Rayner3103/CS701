#!/bin/bash
# Retrain the BP7 trajectory that the 56.49 model is averaged from: all 9,392 labelled images, 2-GPU DDP, global
# batch 16, cosine over 200 epochs (1 warm-up epoch), a checkpoint every 10 epochs. The submission only needs
# epochs 100..180, so the script stops cleanly after epoch 180 (the original run went on to 200: e190/e200 scored
# lower than e180 and were not used). Peak 12.1 GB per GPU with --grad-ckpt; drop it on free 48 GB GPUs (faster,
# same maths). About 13 h on 2 x RTX 6000 Ada when the GPUs are not shared (3 min 40 s per epoch).
# DATA = the converted data from prepare_course_data.py (default: the setup.md layout next to the repo).
#
# The original run was interrupted and resumed three times (memory cap, /tmp full); every resume restored the full
# training_state.pt (model, AdamW, LR schedule, both ranks' RNG), so one uninterrupted run is the same trajectory.
# To resume a stopped run: add --resume "$HERE/runs/BP7_cos200/training_state.pt".
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$HERE/../../../dataset/SARFact-Course-20K}
TORCHRUN=${TORCHRUN:-torchrun}
cd "$HERE/code"
NCCL_P2P_DISABLE=1 $TORCHRUN --standalone --nproc_per_node 2 train.py \
  --data "$DATA" --out "$HERE/runs" --name BP7_cos200 \
  --train-split train+val+test --eval-splits '' --seed 0 --batch-size 16 --workers 1 \
  --epochs 200 --schedule-epochs 200 --warmup-epochs 1 --stop-after-epoch 180 \
  --save-every 10 --save-training-state --reproducible-loader \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 80 --lora-alpha 160 \
  --cascade-rep 1536 --spatial-prior 64 --train-norms --lr 1e-4 --backbone-lr 3e-4 --weight-decay 0.05 \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" \
  --cascade-size-iou --flip-bias -0.3 --grad-ckpt --eval-every 10 --eval-batch-size 4 "$@"
