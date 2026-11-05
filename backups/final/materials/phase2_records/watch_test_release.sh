#!/bin/bash
# Phase 2: wait for test.tar.gz on Hugging Face, download, extract, build the shot-1 candidate (final Phase 1 recipe)
# and a hard-fusion backup, validate the format. NO upload: the user submits manually.
export HF_HOME=/mnt/hdd/junkaichen/hf-cache OMP_NUM_THREADS=4
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python
DATA=/mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data
REPO=/mnt/hdd/junkaichen/CS701/CS701
RUN=/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full9392_cos200_e160to200_s0_20261101_v2/full_train_e200
W=$REPO/.runtime/avg/avg_e100-180.pt
LOG=$REPO/.runtime/phase2/watch.log
cd $REPO
echo "watcher start $(date '+%F %T')" >> $LOG
while true; do
  if $P -c "
from huggingface_hub import HfApi
import sys
files = HfApi().list_repo_files('doem1997/cs701-sar-course-data', repo_type='dataset')
sys.exit(0 if any(f.startswith('test') for f in files) else 1)" 2>>$LOG; then break; fi
  sleep 120
done
echo "test files visible on HF $(date '+%F %T')" >> $LOG
$P -c "
from huggingface_hub import HfApi, hf_hub_download
files = [f for f in HfApi().list_repo_files('doem1997/cs701-sar-course-data', repo_type='dataset') if f.startswith('test')]
print('downloading', files)
for f in files:
    hf_hub_download('doem1997/cs701-sar-course-data', f, repo_type='dataset', local_dir='$DATA')
" >> $LOG 2>&1
cd $DATA && for t in test*.tar.gz; do [ -f "$t" ] && tar -xzf "$t"; done; cd $REPO
TJ=$(ls $DATA/test/images.json 2>/dev/null || ls $DATA/test*/images.json 2>/dev/null | head -1)
TDIR=$(dirname $TJ)
echo "test dir $TDIR; images: $($P -c "import json;print(len(json.load(open('$TJ'))['images']))")" >> $LOG
OUT1=$REPO/submissions/PHASE2_shot1_avg100-180x9_rpn2000_obj0_soft05_cls2_agn_g1_k9.zip
OUT2=$REPO/submissions/PHASE2_backup_avg100-180x9_rpn2000_obj0_soft05_cls2_hardfuse.zip
RAW=$REPO/.runtime/avg/avg_e100-180_test_raw.pt
CUDA_VISIBLE_DEVICES=1 nice -n 10 $P tools/train_gpu_limited.py --memory-fraction 0.15 predict.py --run $RUN --weights $W \
  --images $TDIR --images-json $TJ --batch-size 2 --workers 2 --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --cls-lambda 2 --shift -0.2 \
  --fusion agn --gamma 1.0 --topk 9 --raw-out $RAW --out $OUT1 >> $LOG 2>&1
$P predict.py --run $RUN --from-raw $RAW --images $TDIR --images-json $TJ --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --cls-lambda 2 --shift -0.2 --fusion hard --out $OUT2 >> $LOG 2>&1
for z in $OUT1 $OUT2; do echo "== $z" >> $LOG; $P tools/check_submission.py $z $TJ >> $LOG 2>&1; sha256sum $z | tee -a $LOG > $z.sha256; done
echo "PHASE2 CANDIDATES READY $(date '+%F %T')" >> $LOG
