#!/bin/bash
# Evaluate a finished twin on the val-like holdout: single checkpoints and averaging windows, final post-processing.
# Usage: eval_twin.sh <run_name> <tag>      (raws -> .runtime/phase2/eval_<tag>/, report -> verdict.txt)
set -u
NAME=$1; TAG=$2
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python
S=/tmp/claude-1001/-mnt-hdd-junkaichen-CS701-CS701/52850e2f-f3db-4174-a049-f959babccd69/scratchpad
RUN=/mnt/hdd/junkaichen/CS701/cs701_runs/$NAME/twin
EVAL=/mnt/hdd/junkaichen/CS701/CS701/.runtime/phase2/eval_$TAG; mkdir -p $EVAL
cd /mnt/hdd/junkaichen/CS701/CS701; export OMP_NUM_THREADS=2
# averaging windows (epochs every 5 available from --save-every 5); names used as proxy_eval model names
declare -A WIN
WIN[avg100-180s10]="100 110 120 130 140 150 160 170 180"
WIN[avg100-180s5]="100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180"
WIN[avg100-190s5]="100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190"
WIN[avg120-190s5]="120 125 130 135 140 145 150 155 160 165 170 175 180 185 190"
WIN[avg80-180s10]="80 90 100 110 120 130 140 150 160 170 180"
WIN[avg140-190s5]="140 145 150 155 160 165 170 175 180 185 190"
for k in "${!WIN[@]}"; do
  [ -f $EVAL/$k.pt ] || nice -n 10 $P $S/average_local.py --dir $RUN --epochs ${WIN[$k]} --out $EVAL/$k.pt >> $EVAL/log 2>&1
done
jobs_list=("e130|$RUN/ckpt_e130.pt" "e160|$RUN/ckpt_e160.pt" "e180|$RUN/ckpt_e180.pt" "e190|$RUN/ckpt_e190.pt")
for k in "${!WIN[@]}"; do jobs_list+=("$k|$EVAL/$k.pt"); done
for j in "${jobs_list[@]}"; do
  n=${j%%|*}; wp=${j#*|}
  [ -f $EVAL/${n}_vallike.pt ] && continue
  [ -f $wp ] || { echo "missing $wp" >> $EVAL/log; continue; }
  CUDA_VISIBLE_DEVICES=1 nice -n 10 $P tools/train_gpu_limited.py --memory-fraction 0.12 predict.py --run $RUN --weights $wp --split vallike --batch-size 4 --workers 2 --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1.0 --topk 9 --dump-raw > $EVAL/${n}_vallike.log 2>&1
  mv $RUN/raw_vallike.pt $EVAL/${n}_vallike.pt
done
cd $S
M=""; for j in "${jobs_list[@]}"; do n=${j%%|*}; M="$M $n=$EVAL/${n}"; done
echo "=== $TAG on vallike (992): FINAL post-processing (agn, min_keep 100, snap + wind lattice) ===" > $EVAL/verdict.txt
nice -n 10 $P proxy_eval.py 400 $M --splits vallike --snap --cfg min_keep=100 2>&1 | grep -v "^loading\|^Done\|^$" >> $EVAL/verdict.txt
if $P -c "import torch,sys; r=torch.load('$EVAL/e190_vallike.pt'); sys.exit(0 if r.get('raw_iou') else 1)" 2>/dev/null; then
  for qp in 0.5 1.0 2.0; do
    echo "=== $TAG quality_pow $qp (IoU-branch rescoring) ===" >> $EVAL/verdict.txt
    nice -n 10 $P proxy_eval.py 200 $M --splits vallike --snap --cfg min_keep=100 quality_pow=$qp 2>&1 | grep -v "^loading\|^Done\|^$" >> $EVAL/verdict.txt
  done
fi
echo "EVAL $TAG DONE $(date '+%F %T')" >> $EVAL/log
