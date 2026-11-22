#!/bin/bash
# SC1 seed 1: the F2 recipe at 36 epochs on the local 80:10:10 split (= AA_a4a2's command), scaled to the parameter budget:
# LoRA rank 32 -> 80 (alpha 2x rank), Cascade head width 1024 -> 1536; 124.9M total / 39.7M trainable.
# Same as SC1 with --seed 1 on GPU1; --grad-ckpt saves memory only, same maths and batch 16.           
set -e
RUN=$HOME/cs701_runs/SC1s1_r80h1536
cd $RUN/code
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/mnt/hdd/junkaichen/hf-cache
~/miniconda3/envs/cs701bench/bin/python train.py --data $HOME/cs701_data/dataset/SARFact-Course-20K --out $HOME/cs701_runs --name SC1s1_r80h1536 --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 80 --lora-alpha 160 --cascade-rep 1536 --train-norms --backbone-lr 3e-4 --dump-raw --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --epochs 36 --cascade-size-iou --flip-bias -0.3 --grad-ckpt --seed 1
~/miniconda3/envs/cs701bench/bin/python pp_eval.py --run $RUN
echo "RESULT SC1 done"
