# Report material: SC2 epoch 130 (Δm 51.86)

**English** · [中文](REPORT.zh-CN.md)

> **Purpose: material for the CS701 team report.** This is the best model so far. The body of the method (backbone, Cascade, NWD, A4, A2, post-processing) is the same as F2, whose report material and evolution are in `../44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/REPORT.md`; the local ablation of the scaling is in `../28.90-8020-36ep-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2/REPORT.md`. Reproduction steps are in [README.md](README.md).
>
> Numbers checked on 2026-10-28.

---

## 0. Before writing the report

1. **SC2 changes two things at once**: more parameters (23.8M → 39.7M trainable) and longer training (100 → 150 epochs). The leaderboard's +7.0 Δm cannot be split between them. The evidence that most of it comes from the scaling is the local 80:20 run: at the same 36 epochs, scaling alone gave +5.92 local Δm. The report should state both points.
2. **Epoch 130 was picked by leaderboard score** (both 130 and 150 were submitted; 130 was 0.05 higher). That gap is far below the noise, so they are effectively equal. The report can say plainly "two checkpoints were submitted and the slightly higher one kept"; to avoid any hint of picking checkpoints on val, Phase 2 can use epoch 150 instead.
3. **The trainable budget is used up (39.7M of 40M).** The Terms give credit for fewer trainable parameters at a similar Δm; here parameters bought 7 Δm, and the report's efficiency item should say so.
4. **One seed.** SC2 was trained once. +7.0 is far beyond the run-to-run noise of about ±1 Δm, but say so anyway.
5. **Phase 1 closes on 11-01 at 23:59 SGT**: make sure this submission (or epoch 150) is the posted entry.

## 1. Results

| Submission | Trainable / total | Epochs | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|---|---|
| F1_full (Codabench) | 23.8M / 109.0M | 100 | 41.26 | 95.90 | 46.97 | 71.57 | 96.21 |
| F2 (shift −0.2) | 23.8M / 109.0M | 100 | 44.84 | 95.46 | 48.97 | 71.53 | 95.79 |
| **SC2 epoch 130** | **39.7M / 124.9M** | 130 of 150 | **51.86** | 95.35 | **52.66** | 73.18 | 95.58 |
| SC2 epoch 150 | 39.7M / 124.9M | 150 | 51.81 | 95.44 | 52.61 | 73.24 | 95.72 |

- Against F2: mAP +3.7, AP50 +1.7, F1 −0.1. The whole gain is detection, and mAP rises more than AP50, so the boxes are mainly more precise.
- This matches the local 80:20 run: scaling raised local mAP by 2.5–2.9, and AP on small objects (<16 px) by about 6.
- Epochs 130 and 150 are almost identical: the cosine learning rate is already very low over the last 20 epochs, and the model barely changes.

## 2. What the Terms ask the report to state

| Item | Value |
|---|---|
| Parameters | 124,866,272 total, 39,656,672 trainable (plus 16 floating-point buffer values); breakdown in README section 2 |
| GPU | 2 × NVIDIA RTX 6000 Ada (48 GB), shared with other users |
| Final model training time | 6.38 h (2-GPU DDP, 150 epochs), about 12.8 GPU hours; the epoch-130 checkpoint came at about 5.5 h |
| Peak training memory | 10.6 GB per GPU (gradient checkpointing) |
| Inference time | About 16 ms per image (one GPU, bf16, end-to-end with post-processing, 1,426 val images) |
| Input and preprocessing | As F2: 512×512 gray, each image once |
| Pretrained models | As F2: DINOv3 ViT-B/16 LVD-1689M (timm), no SAR data |
| Seeds | 1 (seed 0) |
| GPU hours of the whole project | About 160 in the F2 report, plus the scaling work (SC1 about 2.6, stopped single-GPU attempts about 2.6, SC2 about 12.8, inference and timing): about 180 |

## 3. How it went

1. **SC1 (local 80:20, 36 epochs)**: same split and epochs, only the parameters scaled; local Δm 37.83 → 43.75. Leaderboard 28.90 (80% of the data and 36 epochs, not comparable with F2).
2. **Full-data training**: single-GPU attempts at 100 and then 150 epochs were stopped early (epochs 8 and 22); the final run trained 150 epochs from scratch on two GPUs.
3. **Checkpoints**: saved every 10 epochs; submissions written for epochs 120, 130, 140 and 150; epochs 130 and 150 were submitted and scored 51.86 and 51.81.

## 4. Next steps

- The budget is used up, so there is no room to widen further. More promising: **redistribute the parameters within the budget**, and add the **S5 spatial prior** that missed F2 (+1.76 local Δm for about 0.2M parameters). See the direction analysis in the conversation.
- Phase 2 allows 2 submissions: predict the test images with epoch 130 (or 150), check the format before submitting, and post afterwards.
