# BP7 fixed architecture: local learning-rate comparison (2026-10-29)

**English** · [中文](README.zh-CN.md)

The user decided to stop architecture exploration, first compare the current learning rate with a half-rate learning rate, and then decide from the results whether to change the training length. This round does not automatically submit online or automatically start full-data retraining.

## Comparison plan fixed in advance

| Item | lr1x | lr05x |
|---|---:|---:|
| Head/neck learning rate | 1e-4 | 5e-5 |
| Backbone trainable-parameter learning rate | 3e-4 | 1.5e-4 |
| Epoch | 130 | 130 |
| Warmup | 1 epoch | 1 epoch |
| Schedule | 130 epoch cosine to 0 | same |
| Seed / global batch | 0 / 16 | same |

Fixed BP7: DINOv3 ViT-B/16, LoRA80/alpha160, Cascade head1536, SpatialPrior64, NWD8, A4 size-IoU, aligned RoI, anchor offset0.5, flip-bias -0.3. 125,076,960 total parameters, 39,867,360 trainable. AdamW weight decay0.05; other augmentation and model config match the original large BP7.

Both groups initialize from generic pretrained DINOv3 with new LoRA/head/prior; old BP7/SC2 weights that have seen all training images are not loaded. GPU0/1 run sequential dual-GPU DDP, workers2 and threads2 per rank, total CPU affinity 4 cores, nice10. Each group has 470 optimizer steps per epoch.

Training uses only local train 7,512 images; local val 940 images are monitored every 10 epochs. The other 940 test images take no part in screening this round; that set has been seen by past experiments and cannot be called a brand-new independent test set. The paths of the three sets are mutually exclusive and combined they match the official train 9,392 images. The official val/test are not used for training or for selection in this round.

The main comparison metric is fixed in advance as **submission post-processing Δm on local val at e130**. Periodic metrics during training use the raw evaluation definition and serve only to monitor curves; the final independent evaluation reuses the classification-fusion lambda1 and full detection post-processing of `predict.py`, with explicit `shift=-0.2`. Local Δm is computed with the exact local reference and cannot be treated as an online score.

## Running and recovery

Run directory: `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_LR130_local_s0_20261029_v2`

- `status.json` / `driver.log`: queue status, lr1x first, then lr05x.
- `launch_manifest.json` / `source.sha256` / `code/`: config, data hashes and frozen code.
- `validation/`: actual dual-GPU continuous-training and interrupt-recovery checks, and the full local evaluation check.
- `lr1x/`, `lr05x/`: each group's training logs, inference weights every 10 epochs, and final evaluation.
- `training_state.pt`: updated atomically every epoch, holding the full model, AdamW, scheduler, per-rank RNG, epoch/global step and metric history.
- `comparison.json`: summarized automatically after both groups complete; differences are defined as half-rate minus baseline.

`--reproducible-loader` rebuilds workers every epoch and fixes random seeds, so that the data order and augmentation random streams of the two groups are paired. The default GPU algorithms stay as they are, and bit-level determinism is not claimed. Checkpoint recovery strictly checks data, config, world size and step count; branch experiments with a different learning rate/schedule are not an ordinary resume and are not implemented in this round.

Before launch: the 7 training-recovery tests, 19 existing ROI tests and 8 independent-evaluation tests passed; the runner's sequential-execution, failure-stop, source-validation and duplicate-launch checks passed. Long training started only after the actual dual-GPU check passed.

Both groups' training and evaluation are expected to take about 11–13 hours, to be corrected by the measured speed of formal training. Full-data retraining and online submissions are arranged only after a worthwhile scheme is selected locally.

The first dual-GPU check record is kept in the directory without `_v2`: independent GPU training already showed parameter numerical differences before recovery, and the original fixed absolute-error gate did not apply. The v2 check verifies consistency of per-rank random state, scheduler/AdamW/global step, model export and training state, and records numerical differences before and after recovery; the architecture, learning rate and training math config are unchanged.

2026-10-29 21:34:38 (Singapore time): the v2 actual dual-GPU recovery check passed. Random state, scheduler and AdamW steps of the two ranks agree, epoch2/global_step4; inference weights match the model in the full state item by item. The maximum parameter differences of independent GPU runs before and after recovery were about 0.000894/0.001289 respectively, recorded, with no claim of bit-level reproduction.

2026-10-29 21:36:23 (Singapore time): the online-definition evaluation check on the full 940 local val images passed, and formal lr1x dual-GPU training started (driver PID 1126688). Step 100/470 of epoch 1 confirmed normal with finite loss; lr05x starts automatically after the first group's training and evaluation complete.

2026-10-29: the user further authorized autonomous experiments. The current two groups keep the original order; another deployed controller will append grouped-learning-rate 130 epochs and the selected recipe at 160 epochs after this queue completes, see `experiments/bp7_lr_epoch_search/README.md`.

## Baseline final score and user online feedback (2026-10-30)

The original-learning-rate 130 epochs completed; the user manually submitted `BP7_local7512_lr1x_e130_val_shift-0.2.zip`, and the feedback screenshot keeps two decimals.

| Metric | Local val (940 images, same post-processing as submission) | Official val (1426 images) |
|---|---:|---:|
| Macro-F1 | 95.32% | 94.80% |
| mAP | 52.27% | 49.66% |
| Accuracy | 94.89% | 95.09% |
| AP50 | 73.12% | 72.07% |

Local Δm is 68.6972 and online Δm is 45.83; the reference baselines and image sets differ, so the two cannot be directly subtracted to measure a regression.

Compared with the old full-data BP7 e130 online 51.76, this Δm is 5.93 lower and mAP 2.87 percentage points lower. The current model trained on 7,512 images and the old model on 9,392; training data, optimization steps and augmentation random streams differ, so the gap cannot all be attributed to any one factor. This is the local screening baseline at the original learning rate; the half-rate and grouped adjustments have no results yet.

Learning rate and epochs continue to be selected by the predefined local validation rules. User online feedback is saved separately, does not change the automatic selection rules, and is not treated as a local validation score. A final stable gain needs paired-seed confirmation, and the final candidate is then retrained on all official training data.

Full online record: `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_LR130_local_s0_20261029_v2/lr1x/submission/leaderboard_score.json`. Submission file SHA256: `08afca47bcb2e6bc5f12f5f7b718e63ccf703b4486a80c4c09a2b2d079da50fe`.

## Final comparison of the two 130-epoch groups (2026-10-30 09:27)

Both groups are complete, uniformly using the same post-processing as submission on local val 940 images:

| Recipe | Local Δm | Macro-F1 | mAP | AP50 |
|---|---:|---:|---:|---:|
| Original learning rate | 68.6972 | 95.3213% | 52.2734% | 73.1243% |
| Overall half rate | 61.4869 | 95.5354% | 48.9887% | 72.4483% |

Half rate relative to baseline: local Δm -7.2103, mAP -3.2847 percentage points, Macro-F1 +0.2141 percentage points, AP50 -0.6761 percentage points. In this same-budget, single-seed comparison, halving overall gives worse detection results; it cannot be inferred from a single run that all lower learning rates/longer schedules are ineffective.

The follow-up controller has passed full-result validation and started hybrid130 (head1e-4, backbone2e-4) at 09:27:51.
