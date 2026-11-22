# BP7 full-data training: 160-epoch cosine, stop and export at e130

**English** · [中文](README.zh-CN.md)

On 2026-10-31 the user authorized with "go ahead" and then asked whether to switch the learning-rate method. This round keeps the original peak learning rate and cosine method, and first verifies the effect of a slower decay on full data.

- Original BP7, DINOv3 LoRA80 / alpha160, Cascade1536, SpatialPrior64, no RoI context.
- All 9392 images of the official train: local train 7512 + val 940 + test 940, with actual paths verified against the official training manifest. Official val 1426 is inference only.
- Retrained from generic pretrained weights; Head LR 1e-4, Backbone LR 3e-4, warmup 1 epoch, seed0, global batch16.
- epochs160, schedule_epochs160, stop_after_epoch130. At e130 the learning rate is about 8.53% of peak.
- Dual-GPU, CPU0–3, nice10, 2 data workers/compute threads per GPU. An inference checkpoint is saved every 10 epochs; the model, optimizer, scheduler and RNG of both GPUs are saved atomically every epoch.
- After e130 completes, 76310 update steps are verified and ckpt_e130.pt exported; training_state.pt is kept for later continuing along the original 160-epoch trajectory. It does not continue to 160 automatically.
- submissions/BP7_full_cos160_e130.zip is generated automatically, with post-processing following the historical config and explicit shift−0.2. Checks format, coverage of 1426 images, finite values, box bounds, at most 100 boxes per image, ZIP integrity and the 25MB limit, and generates SHA256. Submitted manually by the user.

Full-data run directory: /mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full9392_cos160_stop130_s0_20261031.
Controller: tools/run_bp7_full_stop130.py. The directory holds the frozen source, data and source validation, commands, status, smoke-test results, checkpoints and submission validation.

Basis of the choice: for the same e130 submission trained on 7512 images, 130 cosine gave Δ45.83 and 160 cosine gave Δ47.12; a single seed is not enough to prove a stable gain. This run's target is to compare with the original full-data SC2 Δ51.86 and BP7 Δ51.76. After full-data training, the local holdouts already used in training are not evaluated.

Before formal training, one dual-GPU flow smoke test with the original 160-epoch schedule and stop1 is run on 48 official training images, and export is verified on 8 unlabelled official val images. Smoke weights are not used for formal initialization. Failure stops the run and keeps the logs, with no blind retries.

2026-10-31 18:44:33 SGT: e130 and the submission export are complete. Training ran 130 epochs, 76310 updates, taking 6.4355 hours; the full optimizer and dual-GPU RNG are saved. `submissions/BP7_full_cos160_e130.zip` is 9,792,299 bytes, covering 1426 images and 63386 boxes; format, box bounds, SHA256 and ZIP CRC rechecks pass; not yet uploaded. See `checkpoint_validation.json` and `submission_validation.json` in the run directory.

10 new CPU tests pass: training config, real checkpoint compatibility signature, and valid and abnormal cases of submission files.

2026-10-31 12:17:17 SGT: formal training started, controller PID 1543483, torchrun PID 1543839; the dual-GPU smoke test, full checkpoint consistency and the 8-image prediction ZIP validation passed. Epoch1 iter50/587 has been observed, loss 5.236 and finite. Live progress is per status.json and full_train/log.txt in the run directory.

## Official score feedback (2026-10-31)

The user provided a leaderboard screenshot: this `BP7_full_cos160_e130.zip` scored Δm **52.78**, the highest among results fed back in the project so far. The screenshot values have only two decimals; the corresponding submission and weight SHA256 and the historical differences are recorded in `official_result.json`. Whether it has been publicly posted to the leaderboard is unknown.

| Model | Δm | Macro-F1 | mAP | Accuracy | AP50 |
|---|---:|---:|---:|---:|---:|
| Original BP7: 130-epoch cosine, e130 | 51.76 | 95.65 | 52.53 | 95.93 | 73.32 |
| SC2: 150-epoch cosine, e130 | 51.86 | 95.35 | 52.66 | 95.58 | 73.18 |
| This BP7: 160-epoch cosine, e130 | **52.78** | 95.35 | **53.15** | 95.58 | **73.76** |

Compared with the original BP7: Δm +1.02, mAP +0.62 percentage points, AP50 +0.44 percentage points, F1 down 0.30 percentage points; the detection gain offset the classification drop. Compared with the historical best SC2: Δm +0.92, mAP +0.49 percentage points, AP50 +0.58 percentage points, with classification metrics identical at displayed precision. The original BP7 is the more direct training-schedule comparison; SC2 also has configuration differences such as the spatial prior. The single-seed result supports keeping the slower cosine as a candidate, but cannot yet establish a stable or statistically significant gain, nor show that actually training to 160 epochs would be better. This feedback does not change the completed e130 stop state.

2026-10-31 material recheck correction: the old full-data BP7 cos130 did not enable reproducible_loader while this cos160 does, so data order and augmentation random sequences differ. The full-data +1.02 is therefore not strict single-variable causal evidence. The local lr1x cos130 and fixed160 runs do share core code and data manifest and both enable the option; standard local e130 mAP is 51.43→51.99 and F1 95.24→94.92, still single-seed.
