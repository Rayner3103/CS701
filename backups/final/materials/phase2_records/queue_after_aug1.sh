#!/bin/bash
# When AUG1 finishes: (1) start screen #2 (drop-path 0.1, same base recipe) immediately so the GPUs never idle;
# (2) evaluate AUG1 vs the baseline run with averaged windows + final post-processing on both local holdouts.
set -u
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python
S=/tmp/claude-1001/-mnt-hdd-junkaichen-CS701-CS701/52850e2f-f3db-4174-a049-f959babccd69/scratchpad
AUG1=/mnt/hdd/junkaichen/CS701/cs701_runs/AUG1_speckle_gamma_local7512_cos160_s0_20261101
BASE=/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_fixedLR160_local_s0_20261030_direct/schedule160
DP1=/mnt/hdd/junkaichen/CS701/cs701_runs/DP1_droppath01_local7512_cos160_s0_20261102
EVAL=/mnt/hdd/junkaichen/CS701/CS701/.runtime/phase2/eval_aug1
LOG=/mnt/hdd/junkaichen/CS701/CS701/.runtime/phase2/queue.log
mkdir -p $EVAL
until grep -q "train rc=" $AUG1/launch.log 2>/dev/null; do sleep 60; done
echo "AUG1 finished: $(grep 'train rc=' $AUG1/launch.log) at $(date '+%F %T')" >> $LOG
# (1) screen #2: drop-path 0.1 alone (no photometric aug), identical otherwise
mkdir -p $DP1/code && rsync -a $AUG1/code/ $DP1/code/
cat > $DP1/launch.sh <<'EOS'
#!/bin/bash
RUN=/mnt/hdd/junkaichen/CS701/cs701_runs/DP1_droppath01_local7512_cos160_s0_20261102
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python
cd $RUN/code
export NCCL_P2P_DISABLE=1 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/mnt/hdd/junkaichen/CS701/CS701/.runtime/tmp
echo "train start $(date '+%F %T')" >> $RUN/launch.log
nice -n 10 $P -m torch.distributed.run --standalone --nproc_per_node 2 train.py \
  --data /mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K --train-split train --seed 0 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 80 --lora-alpha 160 \
  --cascade-rep 1536 --train-norms --backbone-lr 0.0003 --lr 0.0001 --weight-decay 0.05 --cascade --aligned-roi --rpn-nwd 8 \
  --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3 \
  --grad-ckpt --spatial-prior 64 --warmup-epochs 1 --schedule-epochs 160 --drop-path 0.1 \
  --epochs 160 --stop-after-epoch 150 --save-every 10 --save-training-state --reproducible-loader \
  --batch-size 16 --workers 2 --eval-splits val --eval-every 10 --eval-batch-size 4 \
  --out $RUN --name schedule160 > $RUN/schedule160.stdout.log 2>&1
echo "train rc=$? $(date '+%F %T')" >> $RUN/launch.log
EOS
chmod +x $DP1/launch.sh; setsid nohup $DP1/launch.sh > $DP1/launch.out 2>&1 < /dev/null &
echo "DP1 launched pid $! at $(date '+%F %T')" >> $LOG
sleep 90
# (2) evaluate AUG1 vs baseline: averaged windows e80..e150 (lr 0.5 -> 0.01 on cos160 = analog of e100..e180 on cos200), plus single e150
cd /mnt/hdd/junkaichen/CS701/CS701
for pair in "aug1|$AUG1/schedule160" "base|$BASE"; do
  tag=${pair%%|*}; dir=${pair#*|}
  nice -n 10 $P $S/average_local.py --dir $dir --epochs 80 90 100 110 120 130 140 150 --out $EVAL/${tag}_avg80-150.pt >> $LOG 2>&1
  for split in val test; do
    for w in "${tag}_avg80-150|$EVAL/${tag}_avg80-150.pt" "${tag}_e150|$dir/ckpt_e150.pt"; do
      name=${w%%|*}; wp=${w#*|}
      CUDA_VISIBLE_DEVICES=$([ $split = val ] && echo 0 || echo 1) OMP_NUM_THREADS=2 nice -n 10 $P tools/train_gpu_limited.py --memory-fraction 0.15 predict.py \
        --run $dir --weights $wp --split $split --batch-size 4 --workers 2 --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1.0 --topk 9 --dump-raw \
        > $EVAL/${name}_${split}.log 2>&1
      mv $dir/raw_$split.pt $EVAL/${name}_raw_$split.pt
    done
  done
done
for split in val test; do
  echo "=== $split: avg80-150 aug1 (B) vs base (A) ===" >> $EVAL/verdict.txt
  nice -n 10 $P $S/boot_compare.py $split $EVAL/base_avg80-150_raw_$split.pt $EVAL/aug1_avg80-150_raw_$split.pt 500 >> $EVAL/verdict.txt 2>&1
  echo "=== $split: single e150 aug1 (B) vs base (A) ===" >> $EVAL/verdict.txt
  nice -n 10 $P $S/boot_compare.py $split $EVAL/base_e150_raw_$split.pt $EVAL/aug1_e150_raw_$split.pt 300 >> $EVAL/verdict.txt 2>&1
done
echo "AUG1 VERDICT READY $(date '+%F %T')" >> $LOG
