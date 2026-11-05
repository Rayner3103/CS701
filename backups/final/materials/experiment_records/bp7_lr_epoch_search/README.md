# BP7 fixed architecture: autonomous learning-rate and training-length exploration

**English** · [中文](README.zh-CN.md)

On 2026-10-29 the user authorized designing and continuing training autonomously. The follow-up controller is deployed and waits for the existing learning-rate comparison to complete; this batch appends at most two trainings, expected to run into the evening of 10-30.

Run directory: `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_LR_Epoch_search_20261029`

Predecessor directory: `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_LR130_local_s0_20261029_v2`

## Order and decisions fixed in advance

| Order | Recipe | Head LR | Backbone LR | Epoch | Status/condition |
|---|---|---:|---:|---:|---|
| 1 | Original learning rate | 1e-4 | 3e-4 | 130 | Training in the predecessor queue |
| 2 | Overall half rate | 5e-5 | 1.5e-4 | 130 | Follows automatically in the predecessor queue |
| 3 | Grouped adjustment hybrid130 | 1e-4 | 2e-4 | 130 | Waits for the predecessor to fully complete and release the lock |
| 4 | schedule160 | uses the selected 130-epoch recipe | uses the selected 130-epoch recipe | 160 | Retrained from generic pretraining, cosine period also changed to 160 epochs |

Fixed architecture, augmentation, AdamW/weight decay0.05, warmup 1 epoch, seed0, global batch16. Training on 7,512 images, selection on local val 940 images. All new models initialize from generic pretrained DINOv3 plus random task modules, using no full-data trained weights.

The three 130-epoch groups are compared uniformly by submission post-processing local Δm at e130; recipes within 1.0 of the highest score count as the same tier, and are preferred in the order `lr1x → hybrid130 → lr05x`. The 160-epoch run must be strictly more than 1.0 above the corresponding 130-epoch run to replace the shorter scheme.

**1.0 is an operational trade-off gate, not a statistical-significance standard.** All actual scores and differences are saved. This batch is single-seed exploration; a stable gain still needs paired extra seeds for the candidate and baseline.

The candidate is fixed before looking at local test. Then local test confirmation is done only for the candidate and the original-learning-rate baseline (deduplicated when the same model); hyperparameters are not reselected by test. This test set has been used in past experiments and is not a brand-new independent test set.

## Resources and automatic execution

- Dual-GPU GPU0/1 run serially; 4 CPU cores in total, nice10, 2 threads and 2 workers per rank.
- The follow-up controller must see the predecessor `complete` and acquire the `driver.lock` released by the predecessor before starting GPU jobs.
- Code and data manifests are frozen; the training and evaluation implementation matches the verified predecessor snapshot.
- Verifies full parameters, weight hashes, model budget, 940-image evaluation coverage and fixed post-processing.
- Status updates every 30 seconds. Predecessor failure/death, corrupted results or training failure record the error and stop, with no blind repeated training.
- The full recovery state every epoch and inference weights every 10 epochs continue to be saved.
- This batch does not automatically retrain on full data or upload online submissions.

## Outputs

- `status.json`, `driver.log`: actual status, heartbeat, processes and commands.
- `launch_manifest.json`, `source.sha256`, `code/`: frozen config, selection rules and source.
- `hybrid130/`, `schedule160/`: follow-up training, weights and local val results.
- `lr_selection.json`: the 130-epoch recipe selection and its basis.
- `candidate.json`: the candidate fixed before test confirmation.
- `confirmation_test_*.json`: confirmation results of candidate/baseline.
- `comparison.json`: all validation scores, the 130/160 difference, the candidate's val/test differences relative to baseline, and follow-up paired-seed recommendations.

Before launch, the 19 selection, handoff and result-validation tests passed; independent review confirmed that there is no GPU-overlap path and that strict parameter validation agrees with the current actual run. The controller needs no new GPU smoke test, because the training and evaluation code is file-by-file identical to the snapshot that passed the dual-GPU recovery/full-evaluation checks.

## Morning of 2026-10-30: progress and shared-resource changes

The predecessor completed at 09:27:45 and hybrid130 started automatically at 09:27:51. At the 09:46 check it was in epoch 3 with no training errors, and the queue handoff was normal.

The 09:46 resource check found two extra VLLM inference processes, each taking GPU memory on the order of about 9.8GiB and sharing GPU compute (no intervention). Training-log speed fell from about 28 images/s/rank overnight to about 7.4; if this speed persists, hybrid130 has about 18 hours left and the later 160-epoch run is also pushed back, so the earlier estimate of finishing on the evening of 10-30 no longer applies. If resource use changes, re-estimate. CPU limits, batch, model and training plan keep the original config.


## 2026-10-30 18:04 queue adjustment

The user explicitly chose to try the learning rate raised by 25%, 130 epochs, first. The old controller has been taken over by a new controller, hybrid130 continues to completion; the originally planned schedule160 is postponed and will not start automatically. For the follow-up plan and process handoff record see the [lr125 experiment record](../bp7_lr125/README.md). `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_LR125_local_s0_20261030/status.json` is authoritative for the run status.
