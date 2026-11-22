# F3 (SC2 + S3 capacity split, full data, epochs 120–140 averaged): backup and reproduction

`submission/F3_avg120-140_val_shift-0.2.zip` is Luat's best Phase 1 submission: val Δm **52.56** (mAP **53.03**), 0.70 above SC2 epoch 130 (51.86). The model is SC2's recipe with the trainable budget split differently (screen S3, see [../33.92-8010-36ep-screens-…](../33.92-8010-36ep-screens-B0-S1-S3-S4-S5-DINOv3-Cascade-NWD-SizeIoU-FlipBias-shift0.2/README.md)): a smaller LoRA, the last two ViT blocks fully trained and a narrower Cascade head. It was trained for 150 epochs on all 9,392 labelled images on Colab; the submitted weights are the uniform average of the epoch-120, 130 and 140 checkpoints.

## 1. Results (Phase 1 val, 2026-10-31)

| Submission | Weights | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|---|
| SC2 epoch 130 (Junkai, reference) | 39.7M trainable, epoch 130 | 51.86 | 95.35 | 52.66 | 73.18 | 95.58 |
| F3 epoch 130 | epoch 130 of 150 | 52.38 | 95.43 | 52.91 | 73.05 | 95.72 |
| F3 epoch 140 | epoch 140 | 52.03 | 95.43 | 52.73 | 72.95 | 95.72 |
| F3 epoch 150 | last epoch | 52.01 | 95.43 | 52.72 | 72.97 | 95.72 |
| **F3 avg 120–140 (this backup)** | **mean of ckpt_e120/e130/e140** | **52.56** | 95.34 | **53.03** | **73.15** | 95.65 |
| F3 avg 130–150 | mean of ckpt_e130/e140/model | not submitted | | | | |

- F3 beats SC2 by only +0.25 mAP at epoch 130, although the same change (S3) gave +2.6 official mAP over SC1 in the 36-epoch screens (S3 43.12 vs SC1 ≈ 40.5). **The 36-epoch gain did not survive the 150-epoch schedule**: SC2 catches up as training gets longer. 36-epoch screens should not be used to predict full-run gains of this size.
- Epochs 130–150 differ by 0.2 mAP and share the same classification output (identical F1): training has plateaued; the last epochs slightly overfit (final train loss ≈ 0.04).
- Averaging three neighbouring checkpoints adds +0.12 mAP over the best single one. It is still one network evaluated once per image (no ensemble, no TTA), but check the rules before using it as a final entry; the single-checkpoint fallback is `F3_e130_val_shift-0.2.zip` (52.38).

## 2. Differences from SC2

| Flag | SC2 | F3 | Effect |
|---|---|---|---|
| `--lora-rank` / `--lora-alpha` | 80 / 160 | **48 / 96** | Smaller LoRA (α/r kept at 2), on blocks 0–9 only |
| `--unfreeze-last` | 0 | **2** | The last two ViT blocks are fully trained (backbone lr 3e-4) |
| `--cascade-rep` | 1536 | **1024** | Narrower FC layers in the three Cascade stages |
| Hardware | 2 × RTX 6000 Ada, DDP | **1 × RTX PRO 6000 Blackwell (Colab G4)** | Same global batch 16; 7.43 h, peak 37.8 GB |
| Weights | epoch 130 | **mean of epochs 120/130/140** | +0.12 mAP |

Everything else is identical to SC2: all 9,392 images, cosine schedule over 150 epochs, lr 1e-4 (heads) / 3e-4 (backbone), NWD RPN, 6 px anchors, A4 size-aware Cascade IoU, flip bias −0.3, the final post-processing with the −0.2 px shift.

**Parameters**: 39,152,864 trainable (limit 40M) / 110,194,400 total (limit 130M).

## 3. What is in the folder

| Path | Contents | In git? |
|---|---|---|
| `code/` | Code snapshot used on Colab: SC2's code (51.86 backup) plus `train.py --resume` (last.pt every 2 epochs, atomic save), optional terratorch import (Colab runs Python 3.13), `run_screen.sh` / `run_full.sh` / `smoke*.sh` | yes |
| `code/colab/` | The git-driven Colab worker (`colab_worker.py`), its job queue (`team-queue.json`) and the two worker notebooks (lanes T1/T2) | yes |
| `train_F3_full150.sh` | Retrains F3 (one GPU, 150 epochs, a checkpoint every 10; `--resume` after a dropped runtime) | yes |
| `make_submission_F3_avg120-140.sh`, `avg_checkpoints.py` | Averages the checkpoints, writes and checks the submission | yes |
| `run/args.json`, `run/metrics.json`, `run/log.txt`, `run/F3_full150_run.sh` | train.py arguments, parameter counts, training time, peak memory, full log, the script that ran | yes |
| `submission/` | The five F3 val submissions and their SHA256 | yes |
| `analysis/` | Weakness analysis on the local split (section 5): scripts and their outputs | yes |
| weights | `ckpt_e10 … e150`, `model.pt`, `last.pt` (157 MB each, trainable part only) | no, Luat's Google Drive `MyDrive/cs701-runs/team/F3-S3-full150/` |

Software on Colab: Python 3.13, torch 2.9.0 + cu128, torchvision 0.24.0, timm 1.0.24, pycocotools; no terratorch (TerraMind unavailable, not needed for DINOv3).

## 4. Reproduce

```bash
python code/prepare_course_data.py --src .../cs701-sar-course-data --dst ../../../dataset/SARFact-Course-20K
./train_F3_full150.sh                   # about 7.4 h on a 96 GB GPU; ./train_F3_full150.sh --resume after a crash
./make_submission_F3_avg120-140.sh      # F3_avg120-140_val_shift-0.2.zip
IMAGES=.../cs701-sar-course-data/test ./make_submission_F3_avg120-140.sh    # Phase 2
```

On Colab the same run is the `team-F3-S3-full150` job of `code/colab/team-queue.json` (`bash team/run_full.sh F3-S3-full150 <S3 flags>`), which also writes the e130/e140/e150 zips.

## 5. Where the model loses mAP (local split, S3 = same configuration at 36 epochs)

F3 has no holdout, so the analysis uses the S3 screen (same flags, local train 7,512 images, 36 epochs) on the local val/test (940 + 940 images). Scripts: `analysis/weak.py` (AP by class, IoU threshold and box size), `analysis/bias.py` (systematic box offsets), `analysis/snap.py` (integer-grid snapping). Outputs: `analysis/*.txt`.

| Class (local val) | AP@0.1 | AP50 | AP (0.5:0.95) | median box side |
|---|---|---|---|---|
| bridge | 15.6 | 5.5 | **1.9** | 10 px |
| playground | 24.6 | 16.6 | **7.1** | 15 px |
| wind_turbine | 97.5 | 83.7 | 45.8 | 8.5 px |
| oil_tank | 85.7 | 78.7 | 50.6 | 11 px |
| aircraft | 92.3 | 90.6 | 52.3 | 77 px |
| ship | 88.3 | 81.8 | 39.3 | 26 px |
| car / harbor / airport | 85–96 | 80–95 | 58–62 | |

| Box side | < 8 px | 8–12 | 12–16 | 16–24 | 24–32 | 32–64 | ≥ 64 |
|---|---|---|---|---|---|---|---|
| GT boxes (val) | 601 | 847 | 312 | 308 | 237 | 671 | 347 |
| AP val / test | 20.6 / 13.7 | 21.9 / 22.2 | 29.5 / 27.0 | 37.7 / 41.2 | 48.2 / 48.1 | 55.5 / 59.1 | 49.1 / 49.2 |

1. **Bridge and playground** (≈ 10 mAP lost): even at IoU 0.1 only ~40% of the boxes are found. Earlier work traced this to the labels (these classes come from optically assisted M4-SAR annotations, many objects have no SAR return), so it is a label ceiling rather than a model fault.
2. **High-IoU localisation of tiny boxes**: 44% of the boxes are under 12 px and score AP 13–22. Wind turbines are almost all found (AP50 84–92) but lose half their AP between IoU 0.5 and 0.95: at 8 px a one-pixel error already drops IoU below 0.8. ViT-B/16 features are 16 px cells, larger than the objects.
3. **No systematic error left to post-process**: median centre offsets are 0.1–0.5 px and size errors < 2% (`S3_bias.txt`). Snapping boxes to the integer pixel grid where the GT is integer (aircraft, car, ship, all 256 px images) gives only +0.09 / +0.13 mAP (`S3_snap.txt`).
4. **The spatial prior does not help tiny boxes**: S5 (S3 + `--spatial-prior 64`) has the same AP below 12 px as S3 (`S5_weak.txt`); its prior is added to the neck outputs only, the ViT itself still sees 16 px patches.

Other teams' Phase 1 entries (AP50 77–79 vs our 73, mAP 61–64 vs our 53) are ahead mainly at high IoU, which matches point 2. The next step is architectural: injecting high-resolution CNN features into the ViT blocks (ViT-Adapter-style injector/extractor), a fourth high-IoU Cascade stage or distribution-based box regression, and 14×14 RoI features from stride 4 for small RoIs; trainable parameters can be freed from `--unfreeze-last 2`, which did not pay off at 150 epochs.
