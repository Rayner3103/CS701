# BP7 cos200, parameter average of epochs 100–180, class-agnostic fusion: backup and reproduction

**English** · [中文](README.zh-CN.md)

`submission/BP7_avgE100-180_val_agn.zip` is the team's final Phase 1 entry: val Δm **56.49** on the leaderboard (macro-F1 95.27, mAP 55.10), posted on 2026-11-01 20:59 SGT, rank 2 of 8 when Phase 1 closed. It is one model: the trained tensors of nine checkpoints (epochs 100, 110, …, 180) of one 200-epoch-cosine run on all 9,392 labelled images, averaged into one set of weights, run once per image, with a post-processing chain tuned on the last day of Phase 1.

This README says what is in the folder and how to reproduce it in three ways. Numbers, interpretation and the report items are in [REPORT.md](REPORT.md).

## 1. Results (leaderboard, Phase 1 val, 1,426 images)

| Submission | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|
| F2 (10-27) | 44.84 | 95.46 | 48.97 | 71.53 | 95.79 |
| SC2 e130 (10-28) | 51.86 | 95.35 | 52.66 | 73.18 | 95.58 |
| BP7 cos160 e130 (10-31) | 52.78 | 95.35 | 53.15 | 73.76 | 95.58 |
| BP7 cos200 e180, tuned post-processing (11-01) | 54.83 | 95.09 | 54.29 | 74.09 | 95.44 |
| same weights averaged over e100–180, hard fusion | 56.41 | 95.27 | 55.06 | 74.79 | 95.58 |
| **average e100–180, class-agnostic fusion (this backup)** | **56.49** | **95.27** | **55.10** | **74.85** | **95.58** |

Phase 1 board at the close: Team 9 67.16, **Team 7 (us) 56.49**, Team 2 54.81, Team 1 54.24, Team 8 51.03, Team 3 47.75, Team 5 45.89, Team 4 44.50 (`run/leaderboard_phase1_public_2026-11-01T2120SGT.json`).

## 2. The model

### 2.1 Differences from SC2 (51.86)

| Setting | SC2 | This entry | Effect on val Δm |
|---|---|---|---|
| `--spatial-prior` | off | **64** (a 64-channel CNN branch on the raw image, +210,688 parameters) | part of 51.86 → 52.78 |
| cosine schedule | 150 epochs, e130 taken | **200 epochs** (`--schedule-epochs 200`), checkpoints every 10 | 52.78 → 53.41 at e160 |
| weights used | one checkpoint | **equal average of e100, 110, …, 180** (`average_checkpoints.py`) | +0.6 to +1.0 |
| `--soft-sigma` | 0.75 | **0.5** | +0.29 |
| `--obj-pow` (score × RPN objectness^p) | 0.25 | **0** | +1.05 |
| `--rpn-topn` (test proposals) | 1000 | **2000** | +0.02 |
| `--cls-lambda` (classification fused with detector) | 1 | **2** | +0.04 |
| `--fusion` | none | **agn** (box score = (1 − p_bg) × p_img(class), all 9 classes kept) | +0.5 to +0.7 |
| `--shift` | −0.2 | −0.2 | — |

Everything else is SC2: DINOv3 ViT-B/16 (frozen) + LoRA rank 80 / alpha 160 on `attn.qkv, attn.proj, mlp.fc1, mlp.fc2` + trained LayerNorms, feature pyramid P2–P6, RPN with anchors 6–181 px matched by max(IoU, NWD C=8), 3-stage Cascade R-CNN of width 1536 with size-aware IoU thresholds (A4), aligned RoIAlign, anchor offset 0.5, flip augmentation with `--flip-bias -0.3`, AdamW (head 1e-4, backbone 3e-4, weight decay 0.05), 1 warm-up epoch, 2-GPU DDP with global batch 16, gradient checkpointing (memory only), all 9,392 labelled images (`--train-split train+val+test` of the local split).

### 2.2 Parameters (`run/parameter_breakdown.json`)

| Part | Total | Trainable |
|---|---|---|
| DINOv3 ViT-B/16 backbone (frozen) | 85,135,104 | 0 |
| LoRA r80 on qkv/proj/fc1/fc2, 12 blocks | 11,796,480 | 11,796,480 |
| Backbone LayerNorms (trained) and other 1-D tensors (frozen) | 112,896 | 38,400 |
| Neck: feature pyramid 5,560,000 + spatial prior 210,688 | 5,770,688 | 5,770,688 |
| RPN | 601,645 | 601,645 |
| Cascade RoI heads, 3 stages, width 1536 (+ predictors 64,554) | 21,651,690 | 21,651,690 |
| Image classification head | 8,457 | 8,457 |
| **Total** | **125,076,960** (limit 130M) | **39,867,360** (limit 40M, 99.7%) |

Averaging does not change the architecture or the count: the averaged file holds the same 234 trained tensors (39,867,360 values).

## 3. What is in the folder

| Path | Contents | In git? |
|---|---|---|
| `code/` | Code snapshot: the working tree on 11-01 21:30 (git `8445d58` plus uncommitted additions). Training used the snapshot in the run folder; `run/code_diff_training_snapshot_to_backup.diff` shows the only differences (new opt-in flags, a LingBot backbone file), none of which change this model. | yes |
| `train_BP7_cos200.sh` | Retrains the trajectory on 2 GPUs (stops cleanly after epoch 180, checkpoints every 10) | yes |
| `average_checkpoints.py` | Averages `weights/ckpt_e100..e180.pt` into `weights/model.pt` (bit-identical to the backed-up file) | yes |
| `make_submission.sh` | Writes and checks the submission with the final post-processing; `IMAGES=…/test` for Phase 2; `FROM_RAW=1` needs no GPU | yes |
| `run/args.json`, `run/metrics_e200.json` | All train.py arguments; parameter counts, training time, peak memory | yes |
| `run/logs/` | The five training logs: epoch 1 (original launch, memory error at epoch 2), 2–130, 131–147, 148–160, 161–200 | yes |
| `run/suite_records/` | Launch manifests, checkpoint and submission validations, the decision log of the last day | yes |
| `run/official_scores_phase1.json` | Every official score of the project with date and setting | yes |
| `run/leaderboard_phase1_public_*.json` | The public Phase 1 board 40 minutes before the lock | yes |
| `run/local_holdout_window_sweep.json`, `run/local_holdout_postprocess_bootstrap_{val,test}.json`, `run/boot_eval.py` | The local evidence for the averaging window and the fusion rule, and the paired-bootstrap evaluator that produced it | yes |
| `run/averaging_manifest.json`, `run/parameter_breakdown.json`, `run/dataset_stats.json`, `run/gpu_hours_runs_since_2026-10-28.json`, `run/pip_freeze.txt` | Provenance of the average (SHA256 of every source), parameter table, dataset statistics, GPU-hour estimate, package versions | yes |
| `submission/` | The 56.49 zip (class-agnostic fusion) and the 56.41 zip (hard fusion), with SHA256 | yes |
| `weights/model.pt` | The averaged weights (trained tensors only, 160 MB) | no, local only |
| `weights/ckpt_e100.pt … ckpt_e180.pt` | The nine source checkpoints (9 × 160 MB, hard links to the run folders) | no |
| `weights/val_raw.pt` | Raw model outputs on the 1,426 val images (675 MB); `make_submission.sh FROM_RAW=1` rebuilds the exact zip from it | no |
| `weights/metrics.json` | Copy of the run's metrics.json so that `predict.py --run weights` finds the model arguments | no |
| `report-page.html` | The project report page (bilingual, self-contained HTML; also published as a private claude.ai artifact) | yes |
| `README*.md`, `REPORT*.md` | This guide and the report material, in English and Chinese | yes |

On Junkai's machine `weights/` is at `/mnt/hdd/junkaichen/CS701/CS701/backups/56.49-…/weights/` (hard links into `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full9392_cos200_*`). `weights/SHA256SUMS` lists every file.

## 4. Reproduce

### 4.0 Three ways

| Way | Needs | Time | Result |
|---|---|---|---|
| A. Re-score the raw outputs | `weights/val_raw.pt`, CPU | seconds | the zip, byte-identical |
| B. Predict with the averaged weights | `weights/model.pt`, 1 GPU | about 1 min | the same classification, boxes within GPU float noise |
| C. Retrain, average, predict | 2 GPUs, the data | about 13 h + 1 min | a new trajectory: scores vary by about ±0.5 Δm |

### 4.1 Environment and data

As for F2: sections 4.1–4.2 of `../44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/README.md` (conda env `cs701bench`, torch 2.9.0+cu128, timm 1.0.24; `prepare_course_data.py` converts the Hugging Face download into `dataset/SARFact-Course-20K`). `run/pip_freeze.txt` has the exact package versions. Run the commands below from this folder.

### 4.2 Make the submission (ways A and B)

```bash
# A: no GPU, re-score the saved raw outputs (identical zip contents)
FROM_RAW=1 IMAGES=/path/to/cs701-sar-course-data/val ./make_submission.sh

# B: predict with the averaged weights (1 GPU, about 1 min for 1,426 images)
IMAGES=/path/to/cs701-sar-course-data/val ./make_submission.sh

# Phase 2: the test images (way B; there is no raw file for test until it has run once)
IMAGES=/path/to/cs701-sar-course-data/test ./make_submission.sh
```

The script runs `predict.py --run weights --weights model.pt --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --stage-box mean --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1 --topk 9` and then `tools/check_submission.py`. On a GPU shared with other jobs, prefix with `python tools/train_gpu_limited.py --memory-fraction 0.15 predict.py …` (as in the script's history) to cap memory.

### 4.3 Average the checkpoints

```bash
python average_checkpoints.py                      # weights/ckpt_e100..e180.pt -> weights/model.pt (refuses to overwrite)
python average_checkpoints.py --epochs 100 120 140 160 --out other_window.pt
```

Equal weights 1/9, float64 accumulation, float32 output; `<out>.json` records the SHA256 of every source.

### 4.4 Train from scratch (way C)

```bash
DATA=/path/to/dataset/SARFact-Course-20K ./train_BP7_cos200.sh      # 2 GPUs; stops after epoch 180
python average_checkpoints.py --dir runs/BP7_cos200 --out runs/BP7_cos200/avg_e100-180.pt
cp runs/BP7_cos200/metrics.json weights/ 2>/dev/null; # or point make_submission.sh's RUN at runs/BP7_cos200
```

- The original run: 2 × RTX 6000 Ada shared with other jobs, 13.0 h to epoch 180 (14.14 h to 200), peak 12.1 GB per GPU with `--grad-ckpt`, about 25 images/s per GPU once it had the GPUs to itself (3 min 40 s per epoch). On free 48 GB GPUs drop `--grad-ckpt` for speed; the maths is the same.
- The log should start with `parameters: 125,076,960 total, 39,867,360 trainable`.
- The run stops cleanly after epoch 180 with the learning-rate schedule still the 200-epoch one (`--stop-after-epoch` does not shorten it). The original went on to epoch 200; e190/e200 were not used.
- The original run was interrupted three times (a per-process memory cap too tight for a resumed RPN allocation at epoch 131; `/tmp` full at epoch 147). Each time the complete `training_state.pt` (model, AdamW moments, schedule step, both ranks' RNG) was restored and the following epoch verified, so the trajectory is the same as one uninterrupted run. To resume: `./train_BP7_cos200.sh --resume runs/BP7_cos200/training_state.pt`.
- DDP and cuDNN are not bit-deterministic on GPU: a retrained trajectory gives slightly different weights and scores (about ±0.5 Δm on the leaderboard for a single checkpoint; averaging reduces this).

### 4.5 Checks done on 2026-11-01

- Way A from `weights/val_raw.pt`: `classification.csv` and `detection.json` in the zip are byte-identical to `submission/BP7_avgE100-180_val_agn.zip`.
- `average_checkpoints.py` on the nine backed-up checkpoints: all 234 tensors bit-identical to `weights/model.pt` (the file hash differs only through serialisation).
- Way B with batch 16 on a GPU shared with a training job: classification identical; 59,668 boxes instead of 59,711 (bf16 batch-size effects move a few boxes across the score threshold); 23.2 ms per image.
- Compare the files inside the zips, not the zips' SHA256 (a zip records file times).

### 4.6 Reproduce the local evidence

The averaging window and the fusion rule were checked on an independent trajectory trained on the 7,512-image local split (`cs701_runs/BP7_LR130_local_s0_20261029_v2/lr1x`, checkpoints every 10 epochs) with the two labelled holdouts (940 images each):

```bash
# score one weights file on a labelled local split with the final post-processing (no fusion / hard / agn)
python code/predict.py --run <local run> --weights <ckpt or averaged .pt> --data /path/to/dataset/SARFact-Course-20K \
    --split val --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1 --topk 9
# paired bootstrap of 13 post-processing variants from one raw dump (--dump-raw above writes raw_<split>.pt)
python run/boot_eval.py val <raw_val.pt> 500        # edit ROOT at the top of the file for another machine
```

Results: `run/local_holdout_window_sweep.json` and `run/local_holdout_postprocess_bootstrap_{val,test}.json`; REPORT.md sections 5.2–5.3 summarise them.

### 4.7 Troubleshooting

- `predict.py --run weights` needs `weights/metrics.json` (the model arguments); it is in the folder.
- `--from-raw` ignores `--rpn-topn` (the raw file already holds 2,000 proposals per image).
- A memory error at the start of a resumed run on a shared GPU: raise the memory fraction of `tools/train_gpu_limited.py` (0.2 was too low once the GPUs were ours; 0.35 worked).
- Data-loader workers stalling with "No space left on device": set `TMPDIR` to a disk with space before launching (the 11-01 run hit this at epoch 147).
