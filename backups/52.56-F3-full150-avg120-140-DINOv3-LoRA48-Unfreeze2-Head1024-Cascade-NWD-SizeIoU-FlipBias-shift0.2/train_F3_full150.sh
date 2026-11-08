#!/bin/bash
# Retrain F3: SC2's full-data recipe (all 9,392 labelled images, cosine, batch 16, 150 epochs, a checkpoint every 10)
# with the S3 capacity split: LoRA r48 / alpha 96 on blocks 0-9, the last 2 ViT blocks fully trained, Cascade head
# width 1024 (110.2M total / 39.15M trainable). 7.43 h on one RTX PRO 6000 Blackwell (Colab G4, 96 GB), peak 37.8 GB.
# The backed-up model averages ckpt_e120/e130/e140 (make_submission_F3_avg120-140.sh).
# A dropped runtime continues with: ./train_F3_full150.sh --resume   (last.pt is saved every 2 epochs)
# DATA = the converted data from prepare_course_data.py (default: the setup.md layout next to the repo).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$HERE/../../../dataset/SARFact-Course-20K}
PY=${PY:-python}
cd "$HERE/code"
$PY train.py --data "$DATA" --out "$HERE/runs" --name F3_full150 \
  --train-split train+val+test --eval-splits '' --save-every 10 --epochs 150 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 48 --lora-alpha 96 --unfreeze-last 2 --cascade-rep 1024 \
  --train-norms --backbone-lr 3e-4 --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3 --grad-ckpt "$@"
