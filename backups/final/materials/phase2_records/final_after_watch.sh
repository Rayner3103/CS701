#!/bin/bash
# After the test candidates exist: rebuild shot 1 from the cached raw outputs with min_keep 100 (only appends boxes below
# every thresholded box, so AP cannot drop), then snap to the annotation lattice -> PHASE2_shot1_FINAL.zip. No upload.
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python; R=/mnt/hdd/junkaichen/CS701/CS701; L=$R/.runtime/phase2/watch.log
until grep -q "PHASE2 CANDIDATES READY" $L 2>/dev/null; do sleep 60; done
cd $R; TJ=$(ls /mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/test/images.json 2>/dev/null || ls /mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/test*/images.json | head -1); TDIR=$(dirname $TJ)
RUN=/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full9392_cos200_e160to200_s0_20261101_v2/full_train_e200
MK=submissions/PHASE2_shot1_mk100.zip; FIN=submissions/PHASE2_shot1_FINAL_avg100-180x9_agn_mk100_SNAP.zip
$P predict.py --run $RUN --from-raw .runtime/avg/avg_e100-180_test_raw.pt --images $TDIR --images-json $TJ --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1.0 --topk 9 --min-keep 100 --out $MK >> $L 2>&1
$P tools/snap_submission.py $MK $TJ $FIN >> $L 2>&1 && $P tools/check_submission.py $FIN $TJ >> $L 2>&1 && sha256sum $FIN | tee $FIN.sha256 >> $L
echo "PHASE2 FINAL READY $(date '+%F %T')" >> $L
