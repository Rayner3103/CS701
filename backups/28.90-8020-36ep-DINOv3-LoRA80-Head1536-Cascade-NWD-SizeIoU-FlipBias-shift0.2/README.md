# SC1 (80:20, scaled-up parameters): backup and reproduction

**English** · [中文](README.zh-CN.md)

SC1 is the F2 recipe scaled up to the parameter budget and trained for 36 epochs on the local 80:20 split: local train (7,512 images), evaluated on local val and test (940 each). The 28.90 in the folder name is its Δm on the new leaderboard's Phase 1 val (named by leaderboard score, like the F2 folder); its local Δm is 43.75.

**28.90 is far below F2's 44.84, but that was expected and does not show that scaling failed.** This model differs from F2 mainly in training data and epochs (80% of the data and 36 epochs vs all of it and 100 epochs), not in parameter count. See section 1.2.

This README says what is in the folder and how to reproduce it. Why and how the model was scaled, and how to read the results, are in [REPORT.md](REPORT.md). A full-data run with the same configuration (SC2, 150 epochs) has finished; see section 5 of the REPORT.

## 1. Results

### 1.1 Local holdouts

Same local split as F2's local ablations (seed 0), 36 epochs, final post-processing (no shift). Local Δm is relative to our local reference run E00, averaged over val and test.

| Model | Trainable / total | val F1 / mAP / AP50 | test F1 / mAP / AP50 | Local Δm |
|---|---|---|---|---|
| C36 (37 runs) | 23.8M / 109.0M | 95.39 / 37.46 / — | 95.21 / 38.91 / — | 35.56 ± 0.70 |
| AA (F2's recipe) | 23.8M / 109.0M | 95.05 / 38.43 / 65.85 | 95.55 / 40.01 / 68.27 | 37.83 |
| **SC1 (scaled)** | **39.7M / 124.9M** | **95.65 / 40.96 / 67.96** | 95.05 / **42.87** / **70.25** | **43.75** |
| SC1 − AA | +15.9M | +0.60 / +2.53 / +2.11 | −0.50 / +2.86 / +1.98 | **+5.92** |

- Δm rises by +5.97 on val and +5.87 on test: consistent on both holdouts.
- +5.92 is about 8.5 times the C36 seed standard deviation (0.70). SC1 is a single seed, but a gain this size is unlikely to be luck.
- With a further −0.2 px shift, mAP becomes 41.21 / 43.17 (+0.25 / +0.30), as it did for F2.

### 1.2 Leaderboard (Phase 1 val, 2026-10-27 20:52 SGT, not posted)

| Submission | Training data | Epochs | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|---|---|
| **SC1 (this backup, shift −0.2)** | 80% (7,512 images) | 36 | **28.90** | 94.69 | 40.85 | 67.64 | 94.95 |
| V2, epoch 100 (Codabench) | 90% (8,452 images) | 100 | 38.23 | 95.06 | 45.62 | 70.38 | — |
| F2 (shift −0.2) | 100% (9,392 images) | 100 | 44.84 | 95.46 | 48.97 | 71.53 | 95.79 |

Why it is so far below F2:

- **mAP drives Δm.** The reference mAP is only 26.13, so one mAP point is worth about 1.9 Δm. SC1 is 8 mAP below F2, which costs about 16 Δm.
- **Epochs matter most.** In earlier local runs the same recipe went from 41.5 to 48.9 test mAP between 48 and 100 epochs. SC1 trained 36 epochs, far from the plateau.
- **Local scores overestimate the leaderboard.** SC1's local test mAP is 42.87 against 40.85 on the leaderboard, about 2 points lower; V2 was about 3 lower, so this gap is normal.

The fair test of scaling is AA (F2's recipe, also 36 epochs on 80% of the data). AA was never submitted; with the same gap of about 2 points, its local test mAP of 40.01 suggests about 38 mAP and Δm about 24 on the leaderboard. SC1's 28.90 is about 5 higher, in line with the local +5.92. **This is an estimate, not a measurement.**

Don't post this submission: the posted entry should be the best one (currently F2 at 44.84).

## 2. Differences from the F2 recipe

Three flags change; everything else matches AA (the 36-epoch local version of F2's recipe):

| Flag | AA / F2 | SC1 | Effect |
|---|---|---|---|
| `--lora-rank` / `--lora-alpha` | 32 / 64 | **80 / 160** | 2.5× LoRA capacity, scale α/r kept at 2 |
| `--cascade-rep` | 1024 | **1536** | 1.5× width of the fully connected layers in the three Cascade stages |
| `--grad-ckpt` | off | **on** | Memory only (GPU0 is shared); same maths, batch still 16 |

### Parameters

| Part | F2 trainable | SC1 trainable |
|---|---|---|
| DINOv3 ViT-B/16 backbone (frozen, not trainable) | 0 (85,209,600 total) | 0 (85,209,600 total) |
| LoRA | 4,718,592 | **11,796,480** |
| Backbone LayerNorms | 38,400 | 38,400 |
| Feature pyramid | 5,560,000 | 5,560,000 |
| RPN | 601,645 | 601,645 |
| Cascade RoI heads (3 stages) | 12,878,058 | **21,651,690** |
| Classification head | 8,457 | 8,457 |
| **Trainable total** (limit 40M) | 23,805,152 | **39,656,672** (99.1%) |
| **Total parameters** (limit 130M) | 109,014,752 | **124,866,272** |

## 3. What is in the folder

| Path | Contents | In git? |
|---|---|---|
| `code/` | Code snapshot taken at training time (`cs701_runs/SC1_r80h1536/code`), identical file by file to F2's code (commit `51257a2`) | yes |
| `train_SC1_8020.sh` | Retrains and evaluates SC1 | yes |
| `eval_SC1.sh` | Re-scores local val / test from the saved raw outputs (CPU, minutes) | yes |
| `run/args.json`, `run/metrics.json` | All train.py arguments, parameter counts, training time, peak memory, evaluations every 4 epochs | yes |
| `run/metrics_pp.json` | Local val / test results under the final post-processing (with per-class AP) | yes |
| `run/size_ap_vs_AA.json` | AP by object size, SC1 vs AA | yes |
| `run/log.txt`, `run/SC1_run.sh` | Full training log and the script that ran | yes |
| `run/pip_freeze.txt` | Package versions of the `cs701bench` conda environment | yes |
| `weights/model.pt` | The trained weights (trainable part only, 159 MB) | no, local only |
| `weights/raw_val.pt`, `weights/raw_test.pt` | Raw model outputs on local val / test (204 MB each) | no, local only |
| `submission/` | The leaderboard val submission (`SC1_8020_val_shift-0.2.zip`, 18 MB, 28.90) and its SHA256 | yes |
| `README*.md`, `REPORT*.md` | This guide and the report material, each in English and Chinese | yes |

On Junkai's machine `weights/` is at `/mnt/hdd/junkaichen/CS701/CS701/backups/28.90-8020-36ep-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2/weights/`; the original run folder `/mnt/hdd/junkaichen/CS701/cs701_runs/SC1_r80h1536/` has another copy.

## 4. Reproduce

Environment and data are set up as for F2: sections 4.1–4.2 of `../44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/README.md`. Run the commands below from this folder.

### 4.1 Re-score only (no GPU)

```bash
./eval_SC1.sh     # reads weights/raw_{val,test}.pt, writes weights/metrics_pp.json
```

Checked on 2026-10-27: the re-scored `metrics_pp.json` is identical to `run/metrics_pp.json`.

### 4.2 Train from scratch

```bash
DATA=/path/to/dataset/SARFact-Course-20K ./train_SC1_8020.sh
```

- Original run: one RTX 6000 Ada, 2.64 h, peak memory 12.3 GB (gradient checkpointing on). Speed varied between 13 and 31 images/s because other users' processes shared GPU0.
- With enough free memory (about 45 GB or more), drop `--grad-ckpt`: much faster, same result.
- The log should start with `parameters: 124,866,272 total, 39,656,672 trainable`.
- The script ends by running `pp_eval.py`; compare `runs/SC1_r80h1536/metrics_pp.json` with `run/metrics_pp.json`. DDP, `torch.compile` and cuDNN are not bit-deterministic on GPU, so retrained scores vary (about ±0.35 mAP on C36).

### 4.3 Make the leaderboard submission

```bash
cd code
python predict.py --run ../weights --images /path/to/cs701-sar-course-data/val --shift -0.2 --out SC1_8020_val_shift-0.2.zip
python tools/check_submission.py SC1_8020_val_shift-0.2.zip /path/to/cs701-sar-course-data/val/images.json
```

The original is in `submission/`; to compare zips, compare the `classification.csv` and `detection.json` inside, not the SHA256 of the zips.

### 4.4 Reproduce the AA baseline

Remove `--lora-rank 80 --lora-alpha 160 --cascade-rep 1536 --grad-ckpt` from `train_SC1_8020.sh` and put back `--lora-rank 32 --lora-alpha 64`.
