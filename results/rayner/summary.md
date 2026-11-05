# Training summary: Rayner

Auto-generated from `cs701-sar-benchmark/runs/*/metrics.json`, `.../yolo/*/args.yaml` + `results.csv`, and `gpuClusterResults/`.
Task: 9-class SAR image classification (macro-F1) + object detection (COCO mAP).

## How to read the table

- **Metrics** are `macro-F1 / mAP / Δm` in %, scored on the local split (7,512 train / 940 val / 940 test, seed 0) with the **best checkpoint**
  (chosen on val: best val Δm for `train.py` runs, best ultralytics fitness for YOLO runs). Δm is relative to the course reference
  (ViT + LoRA, 93.31 / 26.13 macro-F1 / mAP) as computed by `train.py`; it is not the leaderboard Δm.
- **Frozen / trainable params**: frozen = total − trainable, from the logged counts. Limits: 130 M total, 40 M trainable.
- **Best epoch**: `train.py` runs: epoch of `best_model.pt` (evaluated every 2 or 4 epochs, so only multiples of those are possible). YOLO runs: epoch with the highest fitness
  (0.1·mAP50 + 0.9·mAP50-95) in `results.csv`, i.e. the epoch of `best.pt`.
- **Epoch**: configured epochs. Early-stopped runs show where they stopped.
- **Augmentation legend**
  - `H+V flip`: the only augmentation of `train.py` (random horizontal and vertical flip, 512×512 resize).
  - `Ultralytics std`: mosaic (off for the last 10 epochs), scale 0.5, translate 0.1, H and V flip (p=0.5), brightness (HSV-V 0.4); no hue/saturation.
  - `rot90 (p=0.75)`: extra random 90/180/270° rotation.
  - `osN` / `paste`: offline small-object oversampling (`augment_small_objects.py`, boxes < 32 px at 512 px): every image with a small box appears N times;
    `paste` adds copy-paste of small boxes (±20 % scale, ±15° rotation) into the extra copies. Originals are always kept; val/test stay unaugmented.
- Seeds: every row is one run; seeds 0, 1, 2 appear as separate rows. Seed noise is about ±1.5 Δm, so smaller differences are not meaningful.

## 1. ViT / TerraMind + Faster R-CNN (course reference code)

The first 7 rows (20 epochs) are the first batch (`run0`); the next 8 (24 epochs) are the re-run (`full_seed0`).
Full fine-tuning and from-scratch variants were not trained locally: those numbers in the repo README are the course's reference values.

| Model | Epoch | Seed | Frozen / trainable params | Data augmentation | Best epoch | Best test (macro-F1 / mAP / Δm) | Best val (macro-F1 / mAP / Δm) |
|---|---:|---:|---:|---|---:|---|---|
| ViT-B/16 (ImageNet-21k) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN | 20 | 0 | 86.0 M / 21.0 M | H+V flip | 20 | 91.4 / 23.5 / -6.1 | 91.6 / 23.0 / -6.9 |
| ViT-B/16 (ImageNet-21k) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN + spatial-prior CNN | 20 | 0 | 86.0 M / 21.2 M | H+V flip | 20 | 92.0 / 25.8 / -1.3 | 91.9 / 24.5 / -3.8 |
| ViT-B/16 (ImageNet-21k) + MoE-LoRA + ViTDet FPN + Faster R-CNN | 20 | 0 | 86.0 M / 21.1 M | H+V flip | 20 | 92.5 / 25.6 / -1.5 | 90.7 / 24.7 / -4.2 |
| ViT-B/16 (ImageNet-21k) + MoE-LoRA + ViTDet FPN + Faster R-CNN + spatial-prior CNN | 20 | 0 | 86.0 M / 21.4 M | H+V flip | 20 | 92.0 / 27.3 / +1.6 | 91.8 / 25.7 / -1.7 |
| TerraMind-1.0-base (EO) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN | 20 | 0 | 85.3 M / 21.0 M | H+V flip | 20 | 91.5 / 25.6 / -2.0 | 89.6 / 24.4 / -5.3 |
| TerraMind-1.0-base (EO) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN + spatial-prior CNN | 20 | 0 | 85.3 M / 21.2 M | H+V flip | 20 | 91.7 / 26.1 / -1.0 | 89.6 / 24.9 / -4.3 |
| TerraMind-1.0-base (EO) + MoE-LoRA + ViTDet FPN + Faster R-CNN | 20 | 0 | 85.3 M / 21.1 M | H+V flip | 20 | 90.4 / 26.7 / -0.5 | 89.1 / 25.2 / -4.0 |
| ViT-B/16 (ImageNet-21k) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN | 24 | 0 | 86.0 M / 21.0 M | H+V flip | 24 | 92.0 / 25.5 / -1.8 | 92.5 / 24.8 / -3.1 |
| ViT-B/16 (ImageNet-21k) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN + spatial-prior CNN | 24 | 0 | 86.0 M / 21.2 M | H+V flip | 24 | 92.0 / 27.9 / +2.7 | 92.0 / 26.3 / -0.3 |
| ViT-B/16 (ImageNet-21k) + MoE-LoRA + ViTDet FPN + Faster R-CNN | 24 | 0 | 86.0 M / 21.1 M | H+V flip | 24 | 91.8 / 27.6 / +2.0 | 90.6 / 26.1 / -1.4 |
| ViT-B/16 (ImageNet-21k) + MoE-LoRA + ViTDet FPN + Faster R-CNN + spatial-prior CNN | 24 | 0 | 86.0 M / 21.4 M | H+V flip | 24 | 92.8 / 28.6 / +4.5 | 91.2 / 26.9 / +0.3 |
| TerraMind-1.0-base (EO) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN | 24 | 0 | 85.3 M / 21.0 M | H+V flip | 24 | 92.4 / 26.8 / +0.7 | 90.2 / 25.6 / -2.7 |
| TerraMind-1.0-base (EO) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN + spatial-prior CNN | 24 | 0 | 85.3 M / 21.2 M | H+V flip | 24 | 91.6 / 27.6 / +1.8 | 90.4 / 26.6 / -0.6 |
| TerraMind-1.0-base (EO) + MoE-LoRA + ViTDet FPN + Faster R-CNN | 24 | 0 | 85.3 M / 21.1 M | H+V flip | 24 | 90.9 / 29.4 / +5.0 | 90.4 / 26.9 / +0.0 |
| TerraMind-1.0-base (EO) + MoE-LoRA + ViTDet FPN + Faster R-CNN + spatial-prior CNN | 24 | 0 | 85.3 M / 21.4 M | H+V flip | 24 | 90.7 / 29.2 / +4.5 | 90.0 / 27.3 / +0.4 |
| TerraMind-1.0-base (EO) + MoE-LoRA + ViTDet FPN + Faster R-CNN | 50 | 0 | 85.3 M / 21.1 M | H+V flip | 50 | 91.8 / 35.1 / +16.4 | 90.2 / 33.6 / +12.7 |
| TerraMind-1.0-base (EO) + MoE-LoRA + ViTDet FPN + Faster R-CNN | 100 (stopped at 96) | 0 | 85.3 M / 21.1 M | H+V flip | 86 | 91.8 / 37.9 / +21.8 | 90.2 / 36.8 / +18.8 |

## 2. DINOv3 ViT-B/16 + Faster R-CNN / Cascade R-CNN

| Model | Epoch | Seed | Frozen / trainable params | Data augmentation | Best epoch | Best test (macro-F1 / mAP / Δm) | Best val (macro-F1 / mAP / Δm) |
|---|---:|---:|---:|---|---:|---|---|
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN | 24 | 0 | 85.2 M / 21.0 M | H+V flip | 24 | 93.5 / 28.2 / +4.1 | 93.0 / 26.4 / +0.4 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN | 24 | 1 | 85.2 M / 21.0 M | H+V flip | 24 | 94.0 / 27.3 / +2.6 | 93.2 / 26.7 / +1.1 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN | 24 | 2 | 85.2 M / 21.0 M | H+V flip | 24 | 93.4 / 27.7 / +3.0 | 93.6 / 26.8 / +1.4 |
| DINOv3 ViT-B/16 (LVD-1689M) + AdaptFormer (bottleneck 64) + ViTDet FPN + Faster R-CNN | 24 | 0 | 85.2 M / 21.3 M | H+V flip | 24 | 90.8 / 26.3 / -1.1 | 89.9 / 25.3 / -3.4 |
| DINOv3 ViT-B/16 (LVD-1689M) + AdaptFormer (bottleneck 256) + ViTDet FPN + Faster R-CNN | 24 | 0 | 85.2 M / 24.8 M | H+V flip | 24 | 92.1 / 27.0 / +0.9 | 90.5 / 25.8 / -2.1 |
| DINOv3 ViT-B/16 (LVD-1689M) + AdaptFormer (bottleneck 64) + LoRA r16 (attn) + ViTDet FPN + Faster R-CNN | 24 | 0 | 85.2 M / 22.2 M | H+V flip | 24 | 94.1 / 27.6 / +3.3 | 93.2 / 26.4 / +0.6 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn+MLP) + ViTDet FPN + Faster R-CNN | 24 | 0 | 85.2 M / 22.5 M | H+V flip | 24 | 93.9 / 28.8 / +5.5 | 93.6 / 27.3 / +2.4 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn) + ViTDet FPN + 3-stage Cascade R-CNN | 24 | 0 | 85.2 M / 20.3 M | H+V flip | 24 | 93.4 / 30.0 / +7.5 | 93.7 / 27.8 / +3.4 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn) + ViTDet FPN + 3-stage Cascade R-CNN | 24 | 1 | 85.2 M / 20.3 M | H+V flip | 24 | 94.2 / 28.7 / +5.5 | 94.0 / 27.7 / +3.4 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn) + ViTDet FPN + 3-stage Cascade R-CNN | 24 | 2 | 85.2 M / 20.3 M | H+V flip | 24 | 93.0 / 29.3 / +5.9 | 93.5 / 27.3 / +2.4 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn+MLP) + ViTDet FPN + 3-stage Cascade R-CNN | 24 | 0 | 85.2 M / 21.8 M | H+V flip | 24 | 93.2 / 30.5 / +8.4 | 93.3 / 29.4 / +6.3 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn) + ViTDet FPN + 3-stage Cascade R-CNN (cascade_confirm re-run) | 24 | 1 | 85.2 M / 20.3 M | H+V flip | 24 | 94.0 / 29.2 / +6.3 | 94.0 / 28.1 / +4.1 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn) + ViTDet FPN + 3-stage Cascade R-CNN (cascade_confirm re-run) | 24 | 2 | 85.2 M / 20.3 M | H+V flip | 24 | 93.0 / 29.0 / +5.3 | 93.4 / 27.7 / +3.1 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn+MLP) + ViTDet FPN + 3-stage Cascade R-CNN (cluster, 100-epoch job) | 100 (log ends at epoch 92; best epoch scored) | 0 | 85.2 M / 21.8 M | H+V flip | 88 | 94.4 / 40.8 / +28.8 | 94.3 / 41.2 / +29.3 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn+MLP) + ViTDet FPN + 3-stage Cascade R-CNN (cluster, 150-epoch job) | 150 (stopped at 96) | 0 | 85.2 M / 21.8 M | H+V flip | 86 | 94.7 / 42.4 / +32.0 | 94.7 / 42.0 / +31.1 |
| DINOv3 ViT-B/16 (LVD-1689M) + LoRA r16 (attn+MLP) + ViTDet FPN + 3-stage Cascade R-CNN | 24 | 0 | 85.2 M / 21.8 M | H+V flip + os3 oversampling (3x small-object images) | 24 | 95.2 / 35.8 / +19.5 | 93.7 / 33.9 / +15.1 |

## 3. DINO-YOLO (YOLO12 / YOLO26 + DINOv3 features fused at P3-P5)

| Model | Epoch | Seed | Frozen / trainable params | Data augmentation | Best epoch | Best test (macro-F1 / mAP / Δm) | Best val (macro-F1 / mAP / Δm) |
|---|---:|---:|---:|---|---:|---|---|
| YOLO26-s only (no DINO), baseline | 24 | 0 | 0.0 M / 10.0 M | Ultralytics std | 24 | 88.2 / 28.3 / +1.4 | 86.6 / 28.2 / +0.4 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO12-s | 24 | 0 | 0.0 M / 31.2 M | Ultralytics std | 24 | 90.4 / 30.8 / +7.4 | 91.1 / 30.7 / +7.6 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO12-s + image-class head | 24 | 0 | 0.0 M / 31.2 M | Ultralytics std | 24 | 94.5 / 29.4 / +6.9 | 94.1 / 28.8 / +5.6 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO12-s | 24 | 0 | 0.0 M / 31.2 M | Ultralytics std + rot90 (p=0.75) | 24 | 89.5 / 27.8 / +1.1 | 90.0 / 28.0 / +1.8 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO12-s | 50 | 0 | 0.0 M / 31.2 M | Ultralytics std | 49 | 92.2 / 34.3 / +15.0 | 92.2 / 33.9 / +14.3 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO12-s | 100 | 0 | 0.0 M / 31.2 M | Ultralytics std | 100 | 92.4 / 36.3 / +19.0 | 92.9 / 37.2 / +20.9 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s | 24 | 0 | 0.0 M / 31.9 M | Ultralytics std | 24 | 90.6 / 31.8 / +9.4 | 91.7 / 31.4 / +9.2 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s | 24 | 1 | 0.0 M / 31.9 M | Ultralytics std | 24 | 91.2 / 31.5 / +9.2 | 92.0 / 31.2 / +9.1 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s | 24 | 0 | 0.0 M / 31.9 M | Ultralytics std + rot90 (p=0.75) | 24 | 90.7 / 29.1 / +4.3 | 90.4 / 29.1 / +4.2 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s with extra P2/4 output | 24 | 0 | 0.0 M / 31.7 M | Ultralytics std | 24 | 91.4 / 30.6 / +7.6 | 91.8 / 30.3 / +7.1 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s | 100 | 0 | 0.0 M / 31.9 M | Ultralytics std | 100 | 93.9 / 37.1 / +21.3 | 93.7 / 37.2 / +21.3 |
| DINOv3 ViT-B/16 (frozen) + YOLO26-s | 24 | 0 | 85.6 M / 10.7 M | Ultralytics std | 22 | 88.6 / 28.5 / +2.1 | 90.3 / 28.4 / +2.8 |
| DINOv3 ViT-B/16 + LoRA r16 (qkv, proj) + YOLO26-s | 24 | 0 | 85.6 M / 11.6 M | Ultralytics std | 24 | 90.3 / 31.5 / +8.7 | 91.0 / 31.1 / +8.2 |
| DINOv3 ViT-B/16 + LoRA r16 (qkv, proj) + YOLO26-s | 24 | 1 | 85.6 M / 11.6 M | Ultralytics std | 24 | 90.8 / 31.1 / +8.1 | 91.4 / 30.7 / +7.8 |
| DINOv3 ViT-B/16 + LoRA r16 (qkv, proj, fc1, fc2) + YOLO26-s | 24 | 0 | 85.6 M / 13.1 M | Ultralytics std | 24 | 90.7 / 31.2 / +8.3 | 91.4 / 31.1 / +8.5 |
| DINOv3 ViT-B/16 + LoRA r64 (qkv, proj) + YOLO26-s | 24 | 0 | 85.6 M / 14.3 M | Ultralytics std | 24 | 91.5 / 30.7 / +7.8 | 91.2 / 30.9 / +8.1 |
| DINOv3 ViT-B/16 + LoRA r256 (qkv, proj) + YOLO26-s | 24 | 0 | 85.6 M / 24.9 M | Ultralytics std | 22 | 91.5 / 31.9 / +10.0 | 91.6 / 31.5 / +9.4 |
| DINOv3 ViT-B/16 + LoRA r256 (qkv, proj) + YOLO26-s | 24 | 1 | 85.6 M / 24.9 M | Ultralytics std | 24 | 92.0 / 31.9 / +10.3 | 92.1 / 31.4 / +9.4 |
| DINOv3 ViT-B/16 + LoRA r256 (qkv, proj) + YOLO26-s | 100 | 0 | 85.6 M / 24.9 M | Ultralytics std | 100 | 93.2 / 37.2 / +21.2 | 93.8 / 37.7 / +22.5 |
| DINOv3 ViT-B/16 (frozen, last 2 blocks trained) + YOLO26-s | 24 | 0 | 71.5 M / 24.9 M | Ultralytics std | 23 | 90.6 / 29.8 / +5.5 | 91.2 / 29.3 / +4.9 |
| DINOv3 ViT-B/16 (frozen, last 4 blocks trained) + YOLO26-s | 24 | 0 | 57.3 M / 39.1 M | Ultralytics std | 24 | 90.8 / 30.5 / +7.0 | 91.8 / 30.3 / +7.2 |
| DINOv3 ViT-B/16 + LoRA r16 (blocks 0-9) + last 2 blocks trained + YOLO26-s | 24 | 0 | 71.5 M / 25.7 M | Ultralytics std | 23 | 91.6 / 31.4 / +9.2 | 91.4 / 30.9 / +8.1 |
| DINOv3 ViT-B/16 + LoRA r16 (qkv, proj) + YOLO26-m | 24 | 0 | 85.6 M / 23.9 M | Ultralytics std | 24 | 91.0 / 31.6 / +9.2 | 91.7 / 31.2 / +8.8 |

## 4. Small-object augmentation on YOLO (24 epochs)

The four arms are compared on DINOv3 ViT-S + YOLO26-s; the unaugmented baseline is `dino_yolo26s_ep24` in section 3 (+9.4 test Δm). The winner (os3) was then repeated on ViT-B/16 + LoRA r256 + YOLO26-s and on plain YOLO26-s (control); the os3 + Cascade R-CNN run is in section 2.

| Model | Epoch | Seed | Frozen / trainable params | Data augmentation | Best epoch | Best test (macro-F1 / mAP / Δm) | Best val (macro-F1 / mAP / Δm) |
|---|---:|---:|---:|---|---:|---|---|
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s | 24 | 0 | 0.0 M / 31.9 M | Ultralytics std + os2 (2x oversampling of small-object images) | 23 | 92.7 / 33.1 / +13.1 | 93.0 / 32.5 / +12.1 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s | 24 | 0 | 0.0 M / 31.9 M | Ultralytics std + os2paste (2x oversampling, copy-paste p=0.5, 1-3 boxes) | 23 | 93.6 / 32.7 / +12.8 | 94.0 / 32.4 / +12.4 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s | 24 | 0 | 0.0 M / 31.9 M | Ultralytics std + os2pasteall (2x oversampling, copy-paste p=0.5, every small box) | 23 | 92.8 / 32.2 / +11.3 | 94.2 / 32.0 / +11.6 |
| DINOv3 ViT-S/16 (fine-tuned) + YOLO26-s | 24 | 0 | 0.0 M / 31.9 M | Ultralytics std + os3 (3x oversampling, no paste) | 24 | 93.6 / 33.8 / +14.8 | 94.2 / 33.1 / +13.9 |
| YOLO26-s only (no DINO) | 24 | 0 | 0.0 M / 10.0 M | Ultralytics std + os3 (3x oversampling, no paste) | 24 | 91.0 / 29.6 / +5.3 | 90.7 / 29.9 / +5.8 |
| DINOv3 ViT-B/16 + LoRA r256 (qkv, proj) + YOLO26-s | 24 | 0 | 85.6 M / 24.9 M | Ultralytics std + os3 (3x oversampling, no paste) | 22 | 94.4 / 33.7 / +15.0 | 95.0 / 33.7 / +15.4 |
| DINOv3 ViT-B/16 + LoRA r256 (qkv, proj) + YOLO26-s (cluster smoke run, not a real result) | 150 (only 3% of train) | 0 | 85.6 M / 24.9 M | Ultralytics std + os3 (3x oversampling, no paste) | 95 | 78.3 / 16.0 / -27.3 | 77.4 / 16.0 / -27.9 |

## Not included

- `smoke/` and `yolo_smoke/`: 1-2 epoch pipeline checks on a data subset.
- `gpuClusterResults/dinov3_pretrained_lora_mlp_cascade_100`: its metrics were rescored into `runs/cluster` (listed in section 2).
