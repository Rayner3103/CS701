#!/bin/bash
# After DP1: evaluate it; then screen #3 IOU1 (IoU-quality branch, with smoke), then screen #4 SA1 (smaller P2 anchors).
Q=/mnt/hdd/junkaichen/CS701/CS701/.runtime/phase2
DP1=/mnt/hdd/junkaichen/CS701/cs701_runs/DP1_droppath01_local7512_cos160_s0_20261102
until grep -q "train rc=" $DP1/launch.log 2>/dev/null; do sleep 120; done
echo "DP1 finished: $(grep 'train rc=' $DP1/launch.log) $(date '+%F %T')" >> $Q/queue.log
$Q/launch_screen.sh IOU1_ioubranch_local7512_cos160_s0_20261102 1 --iou-branch 1.0 > /dev/null 2>&1 &
sleep 60; $Q/eval_screen.sh DP1_droppath01_local7512_cos160_s0_20261102 dp1 >> $Q/queue.log 2>&1
IOU1=/mnt/hdd/junkaichen/CS701/cs701_runs/IOU1_ioubranch_local7512_cos160_s0_20261102
until grep -q "train rc=" $IOU1/launch.log 2>/dev/null; do sleep 120; done
echo "IOU1 finished: $(grep 'train rc=' $IOU1/launch.log) $(date '+%F %T')" >> $Q/queue.log
$Q/launch_screen.sh SA1_smallanchors_local7512_cos160_s0_20261102 1 "--anchor-sizes=5,7,10;16,19,23;32,38,45;64,76,91;128,152,181" > /dev/null 2>&1 &
sleep 60; grep -q "SMOKE_FAILED" $IOU1/launch.log || $Q/eval_screen.sh IOU1_ioubranch_local7512_cos160_s0_20261102 iou1 >> $Q/queue.log 2>&1
SA1=/mnt/hdd/junkaichen/CS701/cs701_runs/SA1_smallanchors_local7512_cos160_s0_20261102
until grep -q "train rc=" $SA1/launch.log 2>/dev/null; do sleep 120; done
echo "SA1 finished: $(grep 'train rc=' $SA1/launch.log) $(date '+%F %T')" >> $Q/queue.log
grep -q "SMOKE_FAILED" $SA1/launch.log || $Q/eval_screen.sh SA1_smallanchors_local7512_cos160_s0_20261102 sa1 >> $Q/queue.log 2>&1
echo "QUEUE2 DONE $(date '+%F %T')" >> $Q/queue.log
