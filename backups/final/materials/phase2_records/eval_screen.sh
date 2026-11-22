#!/bin/bash
# Evaluate a finished local screen against the baseline run on the merged proxy set: avg e80..e150 + single e150.
# Usage: eval_screen.sh <run_name> <tag>
set -u
NAME=$1; TAG=$2
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python
S=/tmp/claude-1001/-mnt-hdd-junkaichen-CS701-CS701/52850e2f-f3db-4174-a049-f959babccd69/scratchpad
RUN=/mnt/hdd/junkaichen/CS701/cs701_runs/$NAME/schedule160
BASE=/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_fixedLR160_local_s0_20261030_direct/schedule160
EVAL=/mnt/hdd/junkaichen/CS701/CS701/.runtime/phase2/eval_$TAG; mkdir -p $EVAL
cd /mnt/hdd/junkaichen/CS701/CS701
export OMP_NUM_THREADS=2
for pair in "$TAG|$RUN" "base|$BASE"; do
  t=${pair%%|*}; dir=${pair#*|}
  [ -f $EVAL/${t}_avg80-150.pt ] || nice -n 10 $P $S/average_local.py --dir $dir --epochs 80 90 100 110 120 130 140 150 --out $EVAL/${t}_avg80-150.pt >> $EVAL/log 2>&1
  for split in val test; do
    for w in "${t}_avg80-150|$EVAL/${t}_avg80-150.pt" "${t}_e150|$dir/ckpt_e150.pt"; do
      n=${w%%|*}; wp=${w#*|}
      [ -f $EVAL/${n}_${split}.pt ] && continue
      CUDA_VISIBLE_DEVICES=1 nice -n 10 $P tools/train_gpu_limited.py --memory-fraction 0.12 predict.py --run $dir --weights $wp --split $split --batch-size 4 --workers 2 --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1.0 --topk 9 --dump-raw > $EVAL/${n}_${split}.log 2>&1
      mv $dir/raw_$split.pt $EVAL/${n}_${split}.pt
    done
  done
done
cd $S && nice -n 10 $P proxy_eval.py 400 base_avg=$EVAL/base_avg80-150 ${TAG}_avg=$EVAL/${TAG}_avg80-150 base_e150=$EVAL/base_e150 ${TAG}_e150=$EVAL/${TAG}_e150 2>&1 | grep -v "^loading\|^Done\|^$" > $EVAL/verdict.txt
echo "EVAL $TAG DONE $(date '+%F %T')" >> $EVAL/log
