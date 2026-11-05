#!/bin/bash
# When the test candidates exist, write snapped versions (integer grid + wind-turbine lattice) next to them.
P=/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python; R=/mnt/hdd/junkaichen/CS701/CS701; L=$R/.runtime/phase2/watch.log
until grep -q "PHASE2 CANDIDATES READY" $L 2>/dev/null; do sleep 60; done
cd $R; TJ=$(ls /mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/test/images.json 2>/dev/null || ls /mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/test*/images.json | head -1)
for z in submissions/PHASE2_shot1_avg100-180x9_rpn2000_obj0_soft05_cls2_agn_g1_k9.zip; do
  o=${z%.zip}_SNAP.zip; $P tools/snap_submission.py $z $TJ $o >> $L 2>&1 && $P tools/check_submission.py $o $TJ >> $L 2>&1 && sha256sum $o | tee $o.sha256 >> $L
done
echo "PHASE2 SNAPPED READY $(date '+%F %T')" >> $L
