# 36-epoch screens on SC2 (B0, S1–S5): backup

Five capacity/architecture screens of the SC2 recipe, each trained for 36 epochs on the local 80:10:10 split (seed 0: 7,512 train / 940 val / 940 test, `prepare_course_data.py`) on Colab G4 (one RTX PRO 6000 Blackwell). B0 is the control (= Junkai's SC1, the 28.90 backup); every other screen changes one thing. S3 won and became F3 ([../52.56-F3-full150-…](../52.56-F3-full150-avg120-140-DINOv3-LoRA48-Unfreeze2-Head1024-Cascade-NWD-SizeIoU-FlipBias-shift0.2/README.md)). The code is that backup's `code/` (`run_screen.sh`, `smoke.sh` are copied here).

## 1. Local results (final post-processing, `metrics_pp.json`)

| Screen | Change on top of B0 | Trainable / total | val F1 / mAP | test F1 / mAP | Δ mAP vs B0 (val / test) | Time |
|---|---|---|---|---|---|---|
| B0 | SC1 recipe (LoRA r80, head 1536) | 39.66M / 124.9M | 95.76 / 40.96 | 94.94 / 42.36 | — | 1.50 h |
| S1 | `--spatial-prior 64` | 39.87M / 125.1M | 95.74 / 41.17 | 95.57 / 42.80 | +0.21 / +0.44 | 1.66 h |
| S2 | `--lr 2e-4 --backbone-lr 6e-4` | 39.66M / 124.9M | cancelled at epoch 20 (val mAP 19.5) | | ≈ −13 | |
| **S3** | **`--lora-rank 48 --lora-alpha 96 --cascade-rep 1024 --unfreeze-last 2`** | 39.15M / 110.2M | **96.21 / 43.02** | 95.36 / **44.57** | **+2.06 / +2.21** | 1.45 h |
| S4 | `--spatial-prior 64 --pretrained-prior --lora-rank 77 --lora-alpha 154` | 39.95M / 125.2M | 95.52 / 41.42 | 95.41 / 42.48 | +0.46 / +0.12 | 1.63 h |
| S5 | S3 + `--spatial-prior 64` | 39.36M / 110.4M | 95.82 / 43.70 | 94.93 / 44.90 | +2.74 / +2.54 | 1.60 h |

- B0 reproduces Junkai's SC1 (val 40.96 / test 42.87) on Colab with Python 3.13 and without terratorch.
- Acceptance bar: mAP up on both val and test, mean ≥ about +1 (one run varies by about ±0.35 mAP per holdout). Only S3 and S5 pass; S5 vs S3 is +0.68 / +0.33, inside the noise.
- S3's gains are in harbor, wind_turbine and airport (per-class AP in `runs/S3/metrics_pp.json`).

## 2. Official results (Phase 1 val, models trained on the 7,512 local-train images only)

| Submission | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|
| SC1 / B0 recipe (Junkai, 28.90 backup) | 28.90 | | ≈ 40.5 (estimated from Δm) | | |
| S3 (`submission/S3_val_shift-0.2.zip`) | 33.50 | 95.18 | 43.12 | 68.29 | 95.51 |
| **S5** (`submission/S5_val_shift-0.2.zip`) | **33.92** | 95.90 | 43.13 | 69.99 | 96.14 |

- The local gain of S3 transfers to the leaderboard (+2.6 mAP official vs +2.1 local).
- The spatial prior adds no mAP officially (S5 = S3 + 0.01); its +0.4 Δm comes from F1 only.
- But the S3 gain shrinks to +0.25 mAP once trained for 150 epochs on all data (F3 vs SC2), see the F3 backup.

## 3. What is in the folder

| Path | Contents |
|---|---|
| `runs/<screen>/log.txt`, `metrics.json`, `metrics_pp.json` | Training log, args + raw local metrics, post-processed local metrics (S2: log only, cancelled) |
| `run_screen.sh` | Copy of the F3 backup's `code/run_screen.sh` (run it from that `code/` folder): `bash run_screen.sh NAME [extra train.py flags]`: SC1 recipe for 36 epochs on local train, then `pp_eval.py` |
| `smoke.sh` | Copy of `code/smoke.sh`, run from the same folder: Builds every screen for 1 epoch on 32 images (the parameter-budget check) |
| `submission/` | The S3 and S5 val submissions (shift −0.2) and their SHA256 |

Raw outputs and `model.pt` of every screen are on Luat's Google Drive (`MyDrive/cs701-runs/team/<screen>/`), not in git.
