# BP7 + classification-only context: 160-epoch local comparison

**English** · [中文](README.zh-CN.md)

On 2026-10-31 the user explicitly asked to stop the current experiment and directly start the first autoresearch candidate. The original 160-epoch training (mid e154, complete checkpoint of e153 kept) and the not-yet-started full-data queue were stopped; no original results were deleted.

## Setup

- Same BP7 source; the architecture config changes by `--roi-context 3 --roi-context-mode classification`.
- The detection RoI classification branch uses enlarged context; box regression and the box updates between Cascade stages still use the local RoI. Head weights are shared. The image classification head is not directly modified.
- Head peak LR 1e-4, Backbone 3e-4, 160 epochs training/cosine, warmup 1 epoch, seed0, train 7512/val 940.
- Starts from generic pretrained weights, not loading trained BP7 weights. The context module changes the random-number consumption of later initialization, so it cannot be claimed that the initial values of the unchanged modules are bit-identical.
- Dual-GPU, batch16, CPU0–3, nice10, 2 workers/threads per rank.
- The three ctx_reduce add 49,344 parameters in total: 125,126,304 total, 39,916,704 trainable, 83,296 trainable parameters left. A dual-GPU 2-step smoke test checks the actual parameters and saved state before formal training starts.

Run directory: `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_ctx3_classification_local160_s0_20261031`. Controller: `tools/run_bp7_context_trial.py`, initial PID 3745643.

## Comparison

The original 160-epoch run was aborted, so comparing the new e160 with the old e130 cannot be called a single-factor ablation. The main comparison is fixed as new context e150 against the saved e150 of the original 160-cos scheme, with the same data split, schedule and seed, and the unified final submission post-processing and shift−0.2. The new e160 is evaluated separately.

This queue does not automatically restart the old training, automatically retrain on full data, or upload official submissions. Results are saved in baseline_e150_post_val.json, context_e150_post_val.json, context_e160_post_val.json and comparison.json.

## Checks

The existing 19 ROI routing/gradient/old-weight compatibility tests and the 19 new controller parameter, budget and result-source validation tests all pass. The new command adds only the two context flags to the original 160 scheme.

The manifest's expected_parameters/expected_trainable_parameters are for the original-implementation consistency check; context_expected_* is the new model's budget. Inference weights are saved every 10 epochs and full state every epoch; failure stops the run with no blind retries.


## 2026-10-31 completed results

Training and evaluation finished at 11:46, with training taking 7.930 hours. All of the following are local val with the unified submission post-processing and cannot be directly compared with official Δm.

| Model | mAP | Macro-F1 | AP50 | Local Δm |
|---|---:|---:|---:|---:|
|Original 160-cos e150|52.6324%|95.5099%|73.1851%|69.6011|
|Context 160-cos e150|53.5982%|95.6948%|73.9320%|71.8566|
|Context final e160|53.6858%|95.6026%|73.9678%|72.0018|

In the matched e150 comparison, local Δm rises by 2.2555 and mAP by about +0.966 percentage points; a single-seed conclusion. The user also asked for an official val submission of the final e160, using the final model.pt and shift−0.2; not yet uploaded.


## Official feedback: e160 submission

2026-10-31 11:50, user screenshot: Δm 46.80, Macro-F1 94.89%, mAP 50.15%, accuracy 95.16%, AP50 72.02%. Saved at the precision shown in the screenshot, with no further decimals inferred.

Compared with the no-context 160-cos e130 submission (47.12/94.58/50.41/94.95/71.62): Δm −0.32, mAP −0.26 percentage points, F1 +0.31, accuracy +0.21, AP50 +0.40 percentage points. This official comparison also changes the checkpoint epoch, so it cannot be attributed to context alone. The local paired e150 advantage did not show up as a lead on the current official e160 submission; small single-seed differences cannot be treated as statistical conclusions. Against the original 130-cos e130 submission's 45.83, the current Δm is +0.97, but that is not a pure context comparison either.

The score and file hash are recorded in the run directory at context160/submission_e160/leaderboard_score.json. No new training or submission has been started on this basis.
