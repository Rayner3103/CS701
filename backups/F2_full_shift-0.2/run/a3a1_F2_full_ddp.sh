#!/bin/bash
# F2 on BOTH GPUs: the F1 recipe (v2 recipe on all 9,392 labelled images, cosine 100 epochs) + A4 (size-aware cascade
# IoU thresholds) + any extra flags in F2_extra_args.txt (B1/A2 if they win their screens); then the Codabench val
# submission. Waits until the other GPU's screen has finished, then up to 30 min for the F2_GO file (flags decided).
Q=$HOME/cs701_runs/queue
until [ -z "$(ls $Q/running/ | grep -v F2_)" ]; do sleep 30; done
for i in $(seq 1 60); do [ -f $Q/F2_GO ] && break; sleep 30; done
EXTRA=$(cat $Q/F2_extra_args.txt 2>/dev/null)
echo "F2 extra flags: $EXTRA"
RUN=$HOME/cs701_runs/F2_full
mkdir -p $RUN
trap 'touch $RUN/FINISHED' EXIT
set -e
rsync -a --exclude .git --exclude docs --exclude __pycache__ --exclude runs --exclude submissions $HOME/CS701/ $RUN/code/
cd $RUN/code
export CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1
~/miniconda3/envs/cs701bench/bin/torchrun --standalone --nproc_per_node 2 train.py \
  --data $HOME/cs701_data/dataset/SARFact-Course-20K --out $HOME/cs701_runs --name F2_full \
  --train-split train+val+test --eval-splits '' --save-every 10 --epochs 100 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 32 \
  --lora-alpha 64 --train-norms --backbone-lr 3e-4 --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou $EXTRA
export CUDA_VISIBLE_DEVICES=0
~/miniconda3/envs/cs701bench/bin/python predict.py --run $RUN --images $HOME/cs701_data/cs701-sar-course-data/val \
  --out $HOME/CS701/submissions/F2_full_val.zip --raw-out $HOME/CS701/submissions/raw/F2_full_val_raw.pt
echo "RESULT F2_full submission written"
