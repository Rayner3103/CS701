# BP7 original learning rate, 160-epoch cosine experiment

**English** · [中文](README.zh-CN.md)

On 2026-10-30 the user asked to keep the original learning rate and try longer training. A separate waiting queue has been started, following the completion and final validation of lr125x, without interrupting the current experiment.

- Head peak LR: 1e-4; Backbone peak LR: 3e-4.
- 160 epochs of training, cosine period 160 epochs, warmup 1 epoch, seed0.
- Reinitialized from generic pretrained weights, not resumed from the 130-epoch checkpoint.
- Fixed architecture, data split 7512 train/940 val, dual-GPU, batch16, CPU0–3, nice10, 2 workers/threads per GPU.
- Original learning-rate baseline: `BP7_LR130_local_s0_20261029_v2/lr1x`.
- New directory: `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_fixedLR160_local_s0_20261030`.
- Controller: `tools/run_bp7_fixed160.py`, using the fixed source snapshot in the new directory; started as PID 3083923.
- Waits for `BP7_LR125_local_s0_20261030` to complete and release the driver lock; on failure it stops, and training is not started overlapping.

## Interpretation of the experiment

In the current implementation the rate decays by cosine after warmup and is 0 at the end of the plan. Changing 130 to 160 changes the whole learning-rate trajectory, so this compares two complete training schemes, and any gain cannot be attributed solely to 30 more epochs. The peak learning rate is unchanged, while the actual mid-run rate is higher and declines more slowly.

Finally, local val metrics with the same submission post-processing are compared with the 130-epoch baseline, recording the differences in mAP, Macro-F1, accuracy, AP50 and local Δm. Exceeding 1.0 local Δm serves only as an operational improvement gate, not statistical significance. A single-seed result cannot prove a stable gain.

This queue runs only one 160-epoch experiment, with no automatic official submission, full-data retraining or test evaluation. `status.json` records status and `comparison.json` the final comparison. The new command differs from the 130-epoch baseline only in the two items `--epochs` and `--schedule-epochs`; the 19 existing scheduling and result-source validation tests pass.

## Adjustment after the user asked to train immediately

Around 21:31 on 2026-10-30, the user asked to train immediately and keep the original learning rate. lr125x (mid epoch 70, complete checkpoint of epoch 69 kept) and the old waiting controller were stopped; after confirming that its training processes had all exited, the separate directory `BP7_fixedLR160_local_s0_20261030_direct` was started, controller PID 3085748. The original waiting directory is kept only as history. The new controller allows the explicitly authorized `stopped_by_user` precondition, but may start only after confirming that the original controller and training processes have exited and the original driver lock is released. Training settings remain Head 1e-4, Backbone 3e-4, 160 epochs training/cosine, warmup 1 epoch; the peak learning rate is unchanged. The status.json in the new directory is authoritative for the current state.


## 2026-10-31 stopped by user

The 160-epoch training stopped mid e154 at the user's request, keeping the complete checkpoint of e153 and the weights of e150; the new context candidate will use e150 of the same 160-cos schedule for a matched comparison. See the [current experiment](../bp7_context160/README.md).
