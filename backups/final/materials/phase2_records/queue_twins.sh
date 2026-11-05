#!/bin/bash
# Twin chain on trainvl/vallike (final recipe): TWIN1 base (running) -> TWIN2 localisation (GIoU 3 + IoU branch 1) -> TWIN3 photometric aug.
Q=/mnt/hdd/junkaichen/CS701/CS701/.runtime/phase2; R=/mnt/hdd/junkaichen/CS701/cs701_runs
wait_done() { until grep -q "train rc=" $1/launch.log 2>/dev/null; do sleep 120; done; echo "$(basename $1) finished: $(grep 'train rc=' $1/launch.log) $(date '+%F %T')" >> $Q/queue.log; }
wait_done $R/TWIN1_base_trainvl8400_cos200_s0_20261101
setsid nohup $Q/launch_twin.sh TWIN2_giou3_iou1_trainvl8400_cos200_s0_20261102 --giou 3.0 --iou-branch 1.0 > $Q/twin2.out 2>&1 < /dev/null &
echo "TWIN2 (giou3+iou1) launched $(date '+%F %T')" >> $Q/queue.log
sleep 120; $Q/eval_twin.sh TWIN1_base_trainvl8400_cos200_s0_20261101 twin1 >> $Q/queue.log 2>&1
wait_done $R/TWIN2_giou3_iou1_trainvl8400_cos200_s0_20261102
setsid nohup $Q/launch_twin.sh TWIN3_aug_trainvl8400_cos200_s0_20261102 --speckle 0.2 --gamma-jitter 0.29 --photo-p 0.5 > $Q/twin3.out 2>&1 < /dev/null &
echo "TWIN3 (aug) launched $(date '+%F %T')" >> $Q/queue.log
sleep 120; $Q/eval_twin.sh TWIN2_giou3_iou1_trainvl8400_cos200_s0_20261102 twin2 >> $Q/queue.log 2>&1
wait_done $R/TWIN3_aug_trainvl8400_cos200_s0_20261102
$Q/eval_twin.sh TWIN3_aug_trainvl8400_cos200_s0_20261102 twin3 >> $Q/queue.log 2>&1
echo "TWIN CHAIN DONE $(date '+%F %T')" >> $Q/queue.log
