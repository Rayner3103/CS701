# Phase 2 overnight plan (written 2026-11-01 21:15 SGT by Claude; user asleep)

**English** · [中文](PLAN.zh-CN.md)

## State
- Phase 1 final: Δm 56.49 = avg(e100..e180 step 10, 9 ckpts of the BP7 cos200 full-data run) + rpn2000, obj_pow 0,
  soft-NMS σ0.5, stage mean, λ=2, shift −0.2, fusion agn γ=1 topk 9. Weights: CS701/.runtime/avg/avg_e100-180.pt.
  Same with fusion hard = 56.41. Previous best 55.41.
- Phase 2: test released 11-02 00:01, 2,065 images, 2 submissions total, deadline 11-08 23:59 SGT. User submits manually.

## Running jobs (all detached with setsid; survive session restarts)
1. cs701_runs/AUG1_speckle_gamma_local7512_cos160_s0_20261101  — local screen of photometric aug
   (speckle max std 0.2 + gamma ±0.29, p=0.5 each) on the exact recipe of BP7_fixedLR160_local_s0_20261030_direct
   (7,512 train, cos160, stop e150, ckpt every 10, eval val every 10). Logs: launch.log, smoke.log, schedule160.stdout.log.
   Baseline for comparison: BP7_fixedLR160_local_s0_20261030_direct/schedule160 (ckpt e10..e150, log.txt has val every 10).
   Decision rule: average ckpts e80..e150 of each run, score both local holdouts with the final post-processing using
   tools/boot_eval.py (paired bootstrap). Win = ≥ +0.3 mAP on both holdouts → full-data cos200 run with aug.
2. CS701/.runtime/phase2/watch_test_release.sh — polls HF for test*, downloads+extracts to cs701_data/cs701-sar-course-data/test,
   builds submissions/PHASE2_shot1_avg100-180x9_rpn2000_obj0_soft05_cls2_agn_g1_k9.zip (recipe above) and
   PHASE2_backup_..._hardfuse.zip, validates format. Log: .runtime/phase2/watch.log. "PHASE2 CANDIDATES READY" when done.
3. Offline post-processing studies (CPU) in the scratchpad boot/ folder: boot_eval2 (γ, σ, min_keep, thr, stage last),
   lambda sweep, stage-weight variants.

## Submission plan
- Shot 1 (morning 11-02): PHASE2_shot1 zip (safe, Phase-1-validated recipe). Post it.
- Shot 2 (by 11-07): best locally-validated challenger (aug-trained full model with the same averaging + post-processing),
  only if it beats the baseline on both local holdouts. Otherwise a refined post-processing of the same weights, or nothing.

## Rules respected
One model, one forward per image at 512, no ensembles/TTA/tiling, val/test only predicted (never trained/labelled/tuned on
without leaderboard feedback), parameter averaging of one trajectory (TA allowed), 125.08M total / 39.87M trainable.
# 2026-11-01 21:36: GPU peer-to-peer (NCCL P2P/CUMEM) hangs on this host; all DDP launches now set NCCL_P2P_DISABLE=1 (verified with a 2-GPU all_reduce test).

## Queue (as of 21:45 SGT)
AUG1 (running, eager, NCCL_P2P_DISABLE=1, ETA ~04:30) -> [queue_after_aug1] DP1 drop-path 0.1 + AUG1 verdict
-> [queue_after_dp1] IOU1 (--iou-branch 1.0, smoke first) + DP1 verdict -> SA1 (P2 anchors 5,7,10) + IOU1 verdict -> SA1 verdict.
Verdicts: .runtime/phase2/eval_<tag>/verdict.txt (merged 1,880-image proxy, uniform + val-like weights, paired bootstrap).
Proxy tooling: scratchpad proxy_eval.py; historical consistency check: scratchpad/proxy/hist_eval.log.

## Post-processing findings on the proxy (W3 local weights, paired bootstrap, 2026-11-01 evening)
- agn (soft fusion, gamma 1, topk 9) > hard > none: +0.09/+0.15 mAP over hard (val/test), official val confirmed +0.04 (56.41 -> 56.49).
- sigma 0.4 vs 0.5 under agn: +0.01/+0.03 (noise-level, consistent sign); min_keep 100: +0.011/+0.015 (P=1.0, monotone);
  sigma 0.4 + min_keep 100 together: +0.022/+0.052 (P .85/.98). Candidate tweak for shot 2 only (shot 1 stays the exact 56.49 recipe).
- No gain: per-class sigma, stage-box weights, gamma 1.5/2, lambda != 2, score_thresh 1e-3, shift != -0.2, last-stage boxes, mult fusion.
- Proxy consistency: local deltas track official deltas (corr 0.98 val-split, 0.85 test-split); val-like class/size reweighting does not improve it.
- Idea queued for after AUG1: 'soup' = average of baseline-run and AUG1-run averaged weights (same seed/data order) -> scratchpad/average_files.py.

## Protocol change 2026-11-01 21:50 (user: do not rely on the old local holdouts; external SAR data is forbidden by the Terms)
- New holdout `vallike` (992 images) = subset of the OLD local val+test matching the official val's (predicted class, size) mix
  (built by scratchpad code; files: dataset/SARFact-Course-20K/{classification/vallike.csv, detection/instances_vallike.json, vallike_split.json}).
  `trainvl` = the other 8,400 official training images.
- Twins = final recipe (cos200, stop e190, ckpt every 5) trained on trainvl, eval on vallike every 10 epochs:
  TWIN1_base_trainvl8400_cos200_s0_20261101 (running since 21:49), then TWIN2_aug_... (queue_twins.sh).
- eval_twin.sh <run> <tag>: windows + singles on vallike with proxy_eval.py --splits vallike.
- Consistency check of vallike vs official scores: 6 historical 7,512 models (never saw vallike) -> scratchpad/proxy/hist_eval.log.
- Old 7,512 screens (AUG1/DP1/IOU1/SA1) and their queues are stopped.

## 2026-11-02 00:50 update
- Phase 2 shot-1 candidate: submissions/PHASE2_shot1_FINAL_avg100-180x9_agn_mk100_SNAP.zip (56.49 recipe + min_keep 100 + lattice snap). Local test-like estimate of the snap+mk100 gain: +0.30 mAP = +0.59 Δm_test [+0.39,+0.80].
- Rejected tonight: stage-probability weights, iterative refinement, w/h rounding, per-cell bias tables, kNN label transfer (rules; user declined).
- Twin chain: TWIN1 base (ETA 07:30 11-02) -> TWIN2 giou3+iou1 -> TWIN3 aug -> [queue_twins2] TWIN4 P2 anchors 5,7,10 -> TWIN5 drop-path 0.1.
  eval_twin.sh now scores with the full FINAL pipeline (--snap, min_keep 100) and quality_pow 0.5/1/2 when an IoU branch exists.
- Time budget: ~10.5 h per run -> about 13 runs fit before 11-08; plan: pick winners by 11-03/04, full-data run(s) of the winner, shot 2 by 11-06.
