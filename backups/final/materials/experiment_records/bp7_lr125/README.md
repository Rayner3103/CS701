# BP7: learning rate raised by 25%, 130 epochs

**English** · [中文](README.zh-CN.md)

On 2026-10-30 the user confirmed trying a higher learning rate first and postponing the originally planned 160-epoch experiment.

| Setting | Original learning-rate baseline | New comparison lr125x |
|---|---:|---:|
| Head LR | 1e-4 | 1.25e-4 |
| Backbone LR | 3e-4 | 3.75e-4 |
| Epoch / cosine schedule | 130 / 130 | 130 / 130 |
| Warmup | 1 epoch | 1 epoch |
| Seed | 0 | 0 |
| Train / local val | 7512 / 940 | 7512 / 940 |

Fixed: BP7 architecture, initialization method, data augmentation, optimizer, global batch16, dual-GPU, CPU0–3, nice10, 2 workers/threads per GPU. The new group initializes from generic pretrained weights and does not resume an existing model. Command arguments other than learning rate match the baseline.

## Queue and handoff

- New directory: `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_LR125_local_s0_20261030`.
- Code: `tools/run_bp7_lr125_trial.py`, launched from the fixed source snapshot in the new directory.
- New controller PID 2665530 started at 18:04 and has entered `waiting_for_hybrid130`.
- Only the old controller PID 1528519 was terminated, keeping torchrun 553814 and training ranks 554080/554081; the old directory's status is marked `superseded`, keeping the original plan and all outputs.
- After hybrid130 ends normally, the complete 130-epoch training state is confirmed and the final unified post-processing validation done, then lr125x starts automatically.
- The original `schedule160` will not start automatically. The new queue starts only this one new training and stops on completion, with no automatic upload, full-data retraining or local test evaluation.

## Comparison and checks

The main comparison is local val metrics at final epoch 130 with the same submission post-processing, against `BP7_LR130_local_s0_20261029_v2/lr1x`. Reports the differences in mAP, Macro-F1, accuracy, AP50 and local Δm; a single-seed gain is not treated as a stable conclusion.

`status.json` records status, `handoff.json` the old process identity and pre-handoff state, `commands.json` the commands actually executed, and `comparison.json` the final comparison. Failure stops the run and keeps the logs; overlapping parallel starts are forbidden.

Checks passed: the 19 existing policy/source validation tests; only the two LR values change between the two commands; process identity and exit detection; in the actual handoff the old controller exited and the original dual-GPU training continued.
