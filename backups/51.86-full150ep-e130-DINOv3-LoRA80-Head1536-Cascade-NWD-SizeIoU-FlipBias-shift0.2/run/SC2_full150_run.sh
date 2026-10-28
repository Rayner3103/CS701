#!/bin/bash
# SC2 (150 epochs, both GPUs): F2's full-data recipe (all 9,392 labelled images, cosine, 2-GPU DDP, global batch 16)
# scaled like SC1 (LoRA r80 / alpha 160, Cascade head width 1536; 124.9M total / 39.7M trainable), 150 epochs with a
# checkpoint every 10. --grad-ckpt because other users hold about 25 GB on each GPU. Then Phase 1 val submissions
# (shift -0.2) for epochs 150, 140, 130, 120.
set -e
RUN=$HOME/cs701_runs/SC2_full150
SUB=$HOME/CS701/submissions
cd $RUN/code
export HF_HOME=/mnt/hdd/junkaichen/hf-cache
PY=~/miniconda3/envs/cs701bench/bin/python
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 ~/miniconda3/envs/cs701bench/bin/torchrun --standalone --nproc_per_node 2 train.py \
  --data $HOME/cs701_data/dataset/SARFact-Course-20K --out $HOME/cs701_runs --name SC2_full150 \
  --train-split train+val+test --eval-splits '' --save-every 10 --epochs 150 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 80 --lora-alpha 160 --cascade-rep 1536 \
  --train-norms --backbone-lr 3e-4 --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3 --grad-ckpt
export CUDA_VISIBLE_DEVICES=0
mkdir -p $SUB/raw
VAL=$HOME/cs701_data/cs701-sar-course-data/val
$PY predict.py --run $RUN --images $VAL --shift -0.2 --out $SUB/SC2_e150_val_shift-0.2.zip --raw-out $SUB/raw/SC2_e150_val_raw.pt
$PY tools/check_submission.py $SUB/SC2_e150_val_shift-0.2.zip $VAL/images.json
for e in 140 130 120; do
  $PY predict.py --run $RUN --weights ckpt_e$e.pt --images $VAL --shift -0.2 --out $SUB/SC2_e${e}_val_shift-0.2.zip
  $PY tools/check_submission.py $SUB/SC2_e${e}_val_shift-0.2.zip $VAL/images.json
done
echo "RESULT SC2_full150 done"
