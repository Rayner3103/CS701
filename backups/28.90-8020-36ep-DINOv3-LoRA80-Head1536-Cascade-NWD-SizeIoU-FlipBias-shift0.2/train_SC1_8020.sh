#!/bin/bash
# Retrain SC1: the F2 recipe at 36 epochs on the local 80:10:10 split (train 7,512; evaluated on local val / test),
# scaled to the parameter budget: LoRA rank 80 / alpha 160, Cascade head width 1536 (124.9M total / 39.7M trainable).
# Same command as the AA_a4a2 baseline apart from --lora-rank / --lora-alpha / --cascade-rep / --grad-ckpt.
# --grad-ckpt only saves memory (peak 12.3 GB instead of about 45 GB); drop it on a free 48 GB+ GPU.
# DATA = the converted data from prepare_course_data.py (default: the setup.md layout next to the repo).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-$HERE/../../../dataset/SARFact-Course-20K}
PY=${PY:-python}
cd "$HERE/code"
$PY train.py --data "$DATA" --out "$HERE/runs" --name SC1_r80h1536 --epochs 36 --dump-raw \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 80 --lora-alpha 160 --cascade-rep 1536 --train-norms --backbone-lr 3e-4 --compile \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" \
  --cascade-size-iou --flip-bias -0.3 --grad-ckpt
$PY pp_eval.py --run "$HERE/runs/SC1_r80h1536"
