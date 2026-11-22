# BP7 automatic length selection and full-data training

**English** · [中文](README.zh-CN.md)

2026-10-31 user authorization: "I'm going to sleep; after you finish the 160 epochs, complete the full train on your own".

Queue `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full_selected_20261031`, controller `tools/run_bp7_full_selected.py`, started as PID 3669143. It waits for `BP7_fixedLR160_local_s0_20261030_direct` to finish final validation and release the lock, then runs automatically; any failure stops it and keeps the state.

## Fixed decision

Compare the original-learning-rate 130 and 160 epochs by final local val Δm with the unified submission post-processing. If 160 is higher by more than 1.0, choose 160; otherwise 130. This gate is operational, not statistical significance. Per the user's authorization it goes directly to full-data training, skips a second-seed recheck, and claims no stable gain. The decision is written to selection.json.

## Full-data training

- Original peak LR fixed: Head 1e-4, Backbone 3e-4.
- The selected epoch count also serves as the cosine period, warmup 1 epoch, seed0.
- BP7 architecture unchanged, restarted from generic pretrained weights, no resuming from locally trained models.
- Local train+val+test total 9392, non-overlapping, verified to match the official training set. Official val 1426 is for prediction only.
- Dual-GPU, batch16, CPU0–3, nice10, 2 workers/threads per GPU.
- Inference weights are saved every 10 epochs and a full checkpoint every epoch. Full-data training no longer evaluates the local val/test already included in training.
- Only one full-data training is run, with no unlimited search and no automatic official upload.

## Outputs on completion

An official val submission is generated automatically from the final model.pt, with the unified post-processing and shift−0.2. Checks format, coverage of 1426 images, box ranges, at most 100 boxes per image, ZIP integrity and size below 25MB, written to submission_validation.json and SHA256. The ZIP is in the repository's submissions, left for the user to submit manually.

Verified: both the 130 and 160 full-data commands parse; learning rate fixed and no resume; the 9392 data union check passes; the 19 existing scheduling/source validation tests pass.


## 2026-10-31 user cancelled this queue

The user asked to stop the current experiment and directly start the classification-context candidate. The full-data queue was stopped in the waiting stage; no full-data training was run. For the current experiment see [BP7 context 160 epochs](../bp7_context160/README.md).
