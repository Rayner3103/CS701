# SC2 epoch 130 (full data, scaled-up parameters): backup and reproduction

**English** · [中文](README.zh-CN.md)

`submission/SC2_e130_val_shift-0.2.zip` is the best Phase 1 submission so far: val Δm **51.86** on the new leaderboard, 7.0 above F2 (44.84). The model is F2's recipe scaled up to the parameter budget (39.7M of 40M trainable), trained for 150 epochs on all 9,392 labelled images, with the epoch-130 checkpoint taken.

This README says what is in the folder and how to reproduce it. Numbers and interpretation for the report are in [REPORT.md](REPORT.md).

## 1. Results (new leaderboard, Phase 1 val, 2026-10-28)

| Submission | Training | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|---|
| F2 (shift −0.2) | 23.8M trainable, 100 epochs | 44.84 | 95.46 | 48.97 | 71.53 | 95.79 |
| **SC2 epoch 130 (this backup)** | **39.7M trainable, epoch 130 of 150** | **51.86** | 95.35 | **52.66** | 73.18 | 95.58 |
| SC2 epoch 150 | same, last epoch | 51.81 | 95.44 | 52.61 | 73.24 | 95.72 |

- 7.0 Δm above F2, all from detection: mAP +3.7, AP50 +1.7, F1 about unchanged.
- Epochs 130 and 150 differ by 0.05 Δm, far below the run-to-run noise (about ±0.5): training had plateaued by about epoch 130.

## 2. Differences from F2

| Flag | F2 | SC2 | Effect |
|---|---|---|---|
| `--lora-rank` / `--lora-alpha` | 32 / 64 | **80 / 160** | 2.5× LoRA capacity, α/r kept at 2 |
| `--cascade-rep` | 1024 | **1536** | 1.5× width of the fully connected layers in the three Cascade stages |
| `--epochs` | 100 | **150** (epoch 130 taken) | Cosine learning rate decays over 150 epochs |
| `--grad-ckpt` | off | **on** | Memory only (another user holds about 25 GB on each GPU); same maths |

Everything else is identical: all 9,392 images, 2-GPU DDP, global batch 16, A4 (`--cascade-size-iou`), A2 (`--flip-bias -0.3`), final post-processing with the −0.2 px shift.

### Parameters

| Part | F2 trainable | SC2 trainable |
|---|---|---|
| DINOv3 ViT-B/16 backbone (frozen) | 0 (85,209,600 total) | 0 (85,209,600 total) |
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
| `code/` | Code snapshot taken at training time, identical file by file to F2's code (commit `51257a2`) | yes |
| `train_SC2_full150.sh` | Retrains (2 GPUs, 150 epochs, a checkpoint every 10) | yes |
| `make_submission_SC2_e130.sh` | Writes and checks the submission with the epoch-130 weights | yes |
| `run/args.json`, `run/metrics.json` | All train.py arguments, parameter counts, training time, peak memory | yes |
| `run/log.txt`, `run/SC2_full150_run.sh` | Full training log and the script that ran | yes |
| `run/pip_freeze.txt` | Package versions of the `cs701bench` conda environment | yes |
| `submission/` | The submission file (51.86) and its SHA256 | yes |
| `weights/model.pt` | The epoch-130 weights (`ckpt_e130.pt`, trainable part only, 159 MB) | no, local only |
| `weights/SC2_e130_val_raw.pt` | Raw model outputs on the Phase 1 val images | no, local only |
| `README*.md`, `REPORT*.md` | This guide and the report material, each in English and Chinese | yes |

On Junkai's machine `weights/` is at `/mnt/hdd/junkaichen/CS701/CS701/backups/51.86-full150ep-e130-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2/weights/`. The full run folder `/mnt/hdd/junkaichen/CS701/cs701_runs/SC2_full150/` keeps every checkpoint from epoch 10 to 140 and the epoch-150 `model.pt`.

## 4. Reproduce

Environment and data are set up as for F2: sections 4.1–4.2 of `../44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/README.md`. Run the commands below from this folder.

### 4.1 Make the submission

```bash
# Predict with the epoch-130 weights (1 GPU, about 1 min)
IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_SC2_e130.sh

# No GPU: re-score the saved raw outputs (seconds)
FROM_RAW=1 IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_SC2_e130.sh

# Phase 2: test images
IMAGES=/path/to/cs701-sar-course-data/test ./make_submission_SC2_e130.sh
```

Checked on 2026-10-28:
- Predicting again with `weights/model.pt`: `classification.csv` and `detection.json` in the zip are byte-identical to those in `submission/`.
- `FROM_RAW=1`: `classification.csv` identical; in `detection.json` boxes differ by at most 0.1 px and scores by at most 0.003, because the raw outputs are stored at lower precision. Negligible for mAP.
- Compare the two files inside the zips, not the SHA256 of the zips (a zip records file times).

### 4.2 Train from scratch

```bash
DATA=/path/to/dataset/SARFact-Course-20K ./train_SC2_full150.sh
```

- Original run: 2 × RTX 6000 Ada, 6.38 h, peak 10.6 GB per GPU (gradient checkpointing on), about 31 images/s per GPU.
- The log should start with `parameters: 124,866,272 total, 39,656,672 trainable`.
- The epoch-130 weights are `runs/SC2_full150/ckpt_e130.pt`. This is a checkpoint partway through a 150-epoch cosine schedule, not the result of "training 130 epochs"; to reproduce it, train with the 150-epoch settings.
- DDP, `torch.compile` and cuDNN are not bit-deterministic on GPU, so retrained scores vary (about ±0.5 Δm on the leaderboard).
- With enough free memory (about 40 GB per GPU or more), drop `--grad-ckpt`: much faster, same result.
