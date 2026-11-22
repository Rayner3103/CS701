#!/bin/bash
# SC2: F2's full-data recipe (all 9,392 labelled images, 100 epochs, cosine) scaled like SC1: LoRA r80 / alpha 160,
# Cascade head width 1536 (124.9M total / 39.7M trainable). One GPU (GPU0 is shared), --grad-ckpt for memory.
# Then the Phase 1 val submission with the -0.2 px shift, as for F2.
set -e
RUN=$HOME/cs701_runs/SC2_full
cd $RUN/code
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/hdd/junkaichen/hf-cache
~/miniconda3/envs/cs701bench/bin/python train.py --data $HOME/cs701_data/dataset/SARFact-Course-20K --out $HOME/cs701_runs --name SC2_full \
  --train-split train+val+test --eval-splits '' --save-every 10 --epochs 100 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 80 --lora-alpha 160 --cascade-rep 1536 \
  --train-norms --backbone-lr 3e-4 --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3 --grad-ckpt
mkdir -p $HOME/CS701/submissions/raw
~/miniconda3/envs/cs701bench/bin/python predict.py --run $RUN --images $HOME/cs701_data/cs701-sar-course-data/val \
  --out $HOME/CS701/submissions/SC2_full_val_shift-0.2.zip --raw-out $HOME/CS701/submissions/raw/SC2_full_val_raw.pt --shift -0.2
~/miniconda3/envs/cs701bench/bin/python tools/check_submission.py $HOME/CS701/submissions/SC2_full_val_shift-0.2.zip $HOME/cs701_data/cs701-sar-course-data/val/images.json
echo "RESULT SC2 done"
