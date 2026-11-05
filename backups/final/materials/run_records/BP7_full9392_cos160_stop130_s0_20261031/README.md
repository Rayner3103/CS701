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

10 new CPU tests pass: training config, real checkpoint compatibility signature, and valid and abnormal cases of submission files.
