#!/bin/bash
# After the first twin chain: TWIN4 small P2 anchors (tiny objects), then TWIN5 drop-path 0.1 (regularisation); each evaluated.
Q=/mnt/hdd/junkaichen/CS701/CS701/.runtime/phase2; R=/mnt/hdd/junkaichen/CS701/cs701_runs
wait_done() { until grep -q "train rc=" $1/launch.log 2>/dev/null; do sleep 120; done; echo "$(basename $1) finished: $(grep 'train rc=' $1/launch.log) $(date '+%F %T')" >> $Q/queue.log; }
until grep -q "TWIN3_aug_trainvl8400_cos200_s0_20261102 finished" $Q/queue.log 2>/dev/null; do sleep 300; done
setsid nohup $Q/launch_twin.sh TWIN4_smallanchors_trainvl8400_cos200_s0_20261103 "--anchor-sizes=5,7,10;16,19,23;32,38,45;64,76,91;128,152,181" > $Q/twin4.out 2>&1 < /dev/null &
echo "TWIN4 (P2 anchors 5,7,10) launched $(date '+%F %T')" >> $Q/queue.log
wait_done $R/TWIN4_smallanchors_trainvl8400_cos200_s0_20261103
setsid nohup $Q/launch_twin.sh TWIN5_droppath01_trainvl8400_cos200_s0_20261103 --drop-path 0.1 > $Q/twin5.out 2>&1 < /dev/null &
echo "TWIN5 (drop-path 0.1) launched $(date '+%F %T')" >> $Q/queue.log
sleep 120; $Q/eval_twin.sh TWIN4_smallanchors_trainvl8400_cos200_s0_20261103 twin4 >> $Q/queue.log 2>&1
wait_done $R/TWIN5_droppath01_trainvl8400_cos200_s0_20261103
$Q/eval_twin.sh TWIN5_droppath01_trainvl8400_cos200_s0_20261103 twin5 >> $Q/queue.log 2>&1
echo "TWIN CHAIN 2 DONE $(date '+%F %T')" >> $Q/queue.log
