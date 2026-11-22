# Report material: F2 (Δm 44.84)

**English** · [中文](REPORT.zh-CN.md)

> **Purpose: material for the CS701 team report.** Sections 1–4 give the numbers, tables and draft wording for every item the "Your report" part of the leaderboard Terms asks for. Section 5 records how the method evolved into F2 and the choice and trade-off at each step. Reproduction steps are in [README.md](README.md).
>
> Numbers checked on 2026-10-27 against `cs701_runs/*/metrics.json`, `metrics_pp.json` and the training logs. The experiments ran from 2026-10-20 to 10-23 on 2 × RTX 6000 Ada.

---

## 0. Before writing the report and submitting

### Do these first

1. **Post F2 on the new leaderboard.** Only https://web-production-c241e.up.railway.app/ counts now; Codabench scores do not.
   - The entry posted at the deadline counts; if nothing is posted, the last successful submission counts.
   - F2 shift −0.2 has been submitted (+44.84) but not posted. Post it from the Team page.
   - After every trial submission, check that the posted entry is still the best one.
2. **The repo is public** (`Rayner3103/CS701`), and the Terms say "Don't share code, data or predictions outside your team".
   - The repo holds all the code, and `backups/.../submission/` holds val predictions.
   - Ask Rayner, the owner, to make it private. Until then, push no prediction files.
3. **Not enough seeds.** The Terms ask for mean ± std over seeds for any gain below about 2 Δm (one run of the reference configuration varies by about ±1 Δm).
   - Now: A4, A2 and A4+A2 (AA) each have one seed; only the C36 baseline has 37 runs.
   - Either run 2–3 more seeds of each (about 2 h per 36-epoch run on one GPU), or state plainly "one seed; the baseline's seed spread is ±0.70 Δm".
   - The table in 4.2 already lists the baseline spread.

### Easy to get wrong in the report

4. **Never mix local Δm and leaderboard Δm.**
   - Local Δm is relative to our own run of the reference model on our local split (E00), which differs from the official split. The leaderboard uses the official reference (val 93.31 / 26.13).
   - Local scores overestimate the leaderboard: V2 was estimated at about +44.8 locally and scored 38.23.
   - List the two separately and name each one's reference.
5. **Post-processing has three versions.** Compare ablations only within one version. The tables in section 4 are grouped by version; don't compare across groups.
6. **Name every pretrained model**, including those used only while exploring (section 3.4).
7. **GPU hours include tuning.** All training runs add up to about 156 GPU hours, 74 of them for 36 seed replicates. Speed tests, inference and failed runs are not counted, so "about 160 GPU hours" is the safe figure.
8. **Say where the code comes from.** It builds on the TA's reference code (`doem97/cs701-sar-benchmark`); we added or rewrote about 2,000 lines (section 3.5). Most of the code was written, and the experiments run, with an AI coding assistant (Claude Code). If the course asks for AI-tool disclosure, state it.
9. **Efficiency earns credit.** At a similar Δm, fewer trainable parameters or less compute scores better in the report grade. We train 23.8M (reference 21.0M, limit 40M), which is worth pointing out.
10. **Upscaling**: 256 px images are stretched to the 512 input. The reference code does the same, and the rule forbids going beyond 512, so this should be fine. If in doubt, ask the TA by email.

### Phase 2 (11-02 00:01 to 11-08 23:59 SGT)

11. Two submissions in total, no reset cards, 2,065 test images.
    - Make the file with `IMAGES=.../cs701-sar-course-data/test ./make_submission_F2.sh`.
    - Check it with `tools/check_submission.py` before submitting, and post it afterwards.
12. The report must say how the Phase 2 submission was chosen; suggested wording in 3.6. If F3 is trained, it replaces F2 only if it beats F2 on both local holdouts and on the Phase 1 leaderboard.

---

## 1. Results

| Submission | Platform | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|---|
| V2, epoch 100 | Codabench val (retired) | 38.23 | 95.06 | 45.62 | 70.38 | — |
| F1_full | Codabench val (retired) | 41.26 | 95.90 | 46.97 | 71.57 | 96.21 |
| **F2_full, shift −0.2** | **New leaderboard, Phase 1 val** | **44.84** | **95.46** | **48.97** | **71.53** | **95.79** |

- The official reference (ViT-B/16 IN-21k + LoRA, mean of 3 seeds) is F1 93.31 / mAP 26.13, Δm = 0.
- On 2026-10-27 the top posted entry on the new leaderboard was Team 9 at +41.66.
- The site states that its scorer gives exactly the numbers Codabench did, so the three rows compare directly.

## 2. Method summary (F2)

```
gray SAR image, stretched to 512×512, normalised with the train-set mean / std
  → DINOv3 ViT-B/16 (frozen) + LoRA r32 on qkv / proj / fc1 / fc2 of all 12 blocks, LayerNorms trained
  → 32×32 tokens of 768
      ├─ mean of the tokens → LayerNorm + Linear → image class (9)
      └─ ViTDet simple feature pyramid (strides 4 / 8 / 16 / 32 / 64)
           → RPN (NWD matching, smallest anchor 6 px)
           → 3-stage Cascade R-CNN (size-dependent IoU thresholds, A4) → boxes, classes, scores
```

- **Training**:
  - All 9,392 labelled images, 100 epochs.
  - AdamW, weight decay 0.05; LoRA lr 3e-4, heads 1e-4; one warmup epoch, then cosine to 0.
  - Batch 16, bf16, `torch.compile`.
  - Flips only, with the A2 flip-bias correction (`--flip-bias -0.3`).
- **Inference**: one forward pass per image, then:
  - Box score × RPN objectness^0.25, per-class soft-NMS (σ 0.75).
  - Boxes are the mean of the three Cascade stages' boxes.
  - Score threshold 1e-5, 30 to 100 boxes per image.
  - Image class = argmax(log p_classifier + log highest detector probability of that class).
  - Every box moved by −0.2 px.

## 3. What the Terms ask the report to state

### 3.1 Parameters (`train.py` prints them)

| Part | Total | Trainable |
|---|---|---|
| DINOv3 ViT-B/16 backbone (frozen) | 85,209,600 | 0 |
| LoRA (r32, 48 linear layers) | 4,718,592 | 4,718,592 |
| Backbone LayerNorms | 38,400 | 38,400 |
| Feature pyramid (neck) | 5,560,000 | 5,560,000 |
| RPN | 601,645 | 601,645 |
| Cascade RoI heads (3 stages) | 12,878,058 | 12,878,058 |
| Classification head | 8,457 | 8,457 |
| **Total** | **109,014,752** (limit 130M) | **23,805,152** (limit 40M) |

There are also 16 floating-point buffer values, which `train.py` counts in its budget check.

### 3.2 Compute

| Item | Value |
|---|---|
| GPU | 2 × NVIDIA RTX 6000 Ada Generation (48 GB) |
| Final model training time | 3.59 h on 2 GPUs (DDP), about 7.2 GPU hours |
| Peak training memory | 38.0 GB per GPU (8 images per GPU) |
| Inference time | 17.7 ms per image on one GPU, bf16, end-to-end average including post-processing (1,426 Phase 1 val images) |
| Whole project | 68 training runs, about 156 GPU hours (74 of them for 36 seed replicates); about 160 with speed tests and inference |

### 3.3 Input and preprocessing

- Each image goes in once at 512×512; no tiling, no TTA.
- Non-square and 256 px images are stretched with `torchvision.transforms.v2.Resize((512, 512))`, as in the reference code. The largest originals are 512 px, so nothing is upscaled beyond 512.
- Read as gray, scaled to [0, 1], normalised with the train-set mean and std (0.2526 / 0.2133).
- timm sums DINOv3's RGB patch-embedding weights into one channel.
- Predicted boxes are mapped back to original image pixels.

### 3.4 Pretrained models (none trained on SAR data)

| Model | Source | Used in |
|---|---|---|
| **DINOv3 ViT-B/16**, LVD-1689M | timm `vit_base_patch16_dinov3.lvd1689m` (Meta, self-supervised on 1.7 B generic web images) | **The final model** and most experiments |
| ViT-B/16, ImageNet-21k | timm `vit_base_patch16_224.augreg_in21k` (reference-code default) | Reference E00, experiments E02, V1 |
| DINOv3 ConvNeXt-T, LVD-1689M | timm `convnext_tiny.dinov3_lvd1689m` | Experiment E07 (full fine-tuning comparison) |

- No teacher, no distillation.
- DINOv2 ViT-B/14, DINO ViT-B/8, DINOv3 ConvNeXt-S/B and ViT-S+ were downloaded into the cache but never used to train a model.

### 3.5 Code

- **External code**:
  - The TA's reference code `doem97/cs701-sar-benchmark`: data loading, ViT + LoRA backbone, base detector and training-script skeleton.
  - PyTorch 2.9 and torchvision 0.24: RPN, AnchorGenerator, RoIAlign, FPN and other operators.
  - timm 1.0.24: backbones and pretrained weights.
  - pycocotools: COCO mAP.
- **What we wrote** (about 2,000 lines added or rewritten on top of the reference code):
  - DINOv3 / ConvNeXt backbones, configurable LoRA (any linear layers, any rank).
  - Cascade R-CNN, NWD anchor matching, size-aware IoU thresholds (A4), flip-bias correction (A2).
  - 2-GPU DDP training, parameter-budget check.
  - Post-processing: soft-NMS, objectness weighting, mean of the stage boxes, shift, detection-informed classification.
  - Raw-output dumps and offline post-processing evaluation (`pp_eval.py`, `pp_study.py`).
  - Submission writing and format checks (`predict.py`, `tools/check_submission.py`), AP by object size.

  Main files: `sarbench/tiny.py`, `sarbench/model.py`, `train.py`, `predict.py`, `sarbench/postprocess.py`.

### 3.6 Validation, and how the Phase 2 submission is chosen

- **Local split**: `prepare_course_data.py` splits the 9,392 labelled images by class with a fixed seed (0) into train 7,512 / val 940 / test 940. Both holdouts are used for evaluation. The official val / test images are only predicted on: never trained on, labelled or tuned on.
- **Local reference**: the local split differs from the official one and trains on only 80% of the data, so we reran the reference model locally (E00) and measure local Δm against it.
- **Acceptance rule**: a change is kept only if it improves both local holdouts and the mean mAP by at least +0.4.
- **Guarding against tuning on the holdouts**: post-processing settings are chosen from a few coarse values (e.g. λ ∈ {0, 0.5, 1}) and must point the same way on 3 models × 2 holdouts.
- **Calibration against the leaderboard**: local scores overestimate the leaderboard, for two reasons: the local split has near-duplicate images, and the class mix differs (the leaderboard val set has more wind turbines and airports, fewer bridges and harbours). Local scores rank methods; the leaderboard gives the absolute level.
- **Final model**: trained on all 9,392 images, no holdout. The epoch count (100) comes from V2, which trained 120 epochs on local train + val and was monitored on local test: it plateaued after epoch 90 and peaked at epoch 100.
- **Phase 2 choice (suggested wording)**: Phase 2 uses the best model on the Phase 1 leaderboard (currently F2). A new model replaces it only if it beats F2 on both local holdouts and on the Phase 1 leaderboard.

### 3.7 Seeds

- **Baseline**: the C36 recipe (36 epochs, local train) has 37 runs, C36 itself plus 36 seeds. Under the final post-processing (shift −0.4):

| | macro-F1 | mAP |
|---|---|---|
| Local val | 95.39 ± 0.29 | 37.46 ± 0.35 |
| Local test | 95.21 ± 0.36 | 38.91 ± 0.36 |
| Local Δm (mean of val and test) | 35.56 ± 0.70 | |

- **Every other experiment is one seed**, including A4, A2, AA and the final model. See item 3 of section 0.

## 4. Ablations (local holdouts)

**Local Δm**: relative to our local run of the reference model E00 without post-processing (val F1 91.50 / mAP 22.41, test 91.61 / 23.31), averaged over val and test.

### 4.1 First round: 24 epochs (post-processing v1: soft-NMS + objectness + detection-informed classification, threshold 0.001)

| Run | Change from the previous step | val F1 / mAP | test F1 / mAP | Local Δm (no post-proc.) | Local Δm (v1) |
|---|---|---|---|---|---|
| E00 | Reference: ViT-B/16 IN21k + LoRA r16 (attention only) | 92.45 / 23.70 | 92.67 / 25.12 | 0 | 3.9 |
| E01 | DINOv3 backbone | 93.82 / 26.48 | 93.60 / 28.12 | 8.1 | 10.9 |
| E02 | IN21k, LoRA r32 on every linear layer + LayerNorms | 93.64 / 28.15 | 94.10 / 30.06 | 11.4 | 14.9 |
| **E04r** | DINOv3 + the E02 recipe | 95.21 / 31.10 | 94.61 / 32.52 | 18.1 | 21.4 |
| D1 | E04r + Cascade R-CNN | 95.46 / 31.95 | 95.38 / 33.20 | 21.0 | 23.4 |
| D2 | E04r + NWD matching and 6 px anchors | 95.11 / 32.03 | 95.01 / 32.92 | 19.7 | 23.0 |
| D6 | E04r, LoRA rank 64 | 94.86 / 32.17 | 94.81 / 32.03 | 19.0 | 22.0 |
| D3 | E04r + CNN spatial prior + multi-level neck | 94.33 / 26.17 | 94.26 / 26.43 | 5.1 | 9.0 |
| D4 | E04r + D4 rotations + copy-paste | 94.72 / 26.06 | 95.00 / 27.00 | 7.1 | 9.8 |
| E07 | DINOv3 ConvNeXt-T, full fine-tuning (35.5M) | 94.75 / 25.84 | 94.55 / 26.64 | 6.8 | 9.1 |
| E09b | DINOv3 + LoRA r64 + drop path + LoRA+ and more | 95.06 / 25.60 | 94.68 / 26.76 | 6.2 | 9.1 |

F1 and mAP are after post-processing v1.

### 4.2 Second round: 36 / 48 epochs (final post-processing v3)

| Run | Change on top of C36 | val F1 / mAP | test F1 / mAP | Local Δm | vs C36 mean |
|---|---|---|---|---|---|
| L36 | E04r for 36 epochs (no Cascade / NWD) | 94.79 / 36.42 | 95.43 / 37.49 | 32.77 | −2.79 |
| **C36** (37 runs) | E04r + Cascade + NWD, 36 epochs | 95.39 / 37.46 | 95.21 / 38.91 | **35.56 ± 0.70** | 0 |
| C48 | same, 48 epochs | 95.23 / 40.96 | 95.10 / 41.60 | 42.28 | +6.72 |
| **AA = A4 + A2 (F2's combination)** | both, no shift | 95.05 / 38.43 | 95.55 / 40.01 | **37.83** | **+2.27** |
| A4 | Size-aware Cascade IoU | 95.27 / 38.15 | 95.27 / 39.78 | 37.25 | +1.69 |
| A2 | Flip-bias correction, no shift | 95.74 / 37.86 | 95.00 / 39.61 | 36.79 | +1.23 |
| S5 | CNN spatial prior | 95.39 / 38.23 | 95.04 / 39.79 | 37.32 | +1.76 |
| N1 | Last 2 blocks fully trained (37.2M trainable) | 95.05 / 38.05 | 95.21 / 39.69 | 36.97 | +1.41 |
| A4b | Relaxed IoU below 16 px only | 95.94 / 37.82 | 95.02 / 39.41 | 36.60 | +1.04 |
| R1 | NWD constant 16 | 95.64 / 38.05 | 95.39 / 39.08 | 36.52 | +0.96 |
| S6 | Trained patch embedding | 94.98 / 37.62 | 95.23 / 39.39 | 36.16 | +0.60 |
| E1 | EMA | 95.98 / 37.40 | 95.02 / 38.74 | 35.43 | −0.13 |
| B1 | IoU-quality branch | 95.27 / 37.25 | 95.01 / 38.75 | 35.08 | −0.48 |
| S8 | Overlapping patches (stride 8, 2.6× compute) | 94.91 / 37.09 | 95.59 / 38.40 | 34.58 | −0.98 |
| R2 | 1024 sampled RoIs | 95.91 / 35.38 | 95.62 / 37.08 | 31.55 | −4.01 |

- The C36 row is the mean of 37 runs.
- Final post-processing v3 = v1 + mean of the stage boxes + score threshold 1e-5 + at least 30 boxes per image + shift −0.4. A2 and AA correct the bias in training and are not shifted.
- Only AA clears 3 standard deviations of the C36 mean (3 × 0.70 = 2.1). A4, S5, A2 and N1 sit at 2–2.5 standard deviations from a single seed each; handle them as in item 3 of section 0.

### 4.3 Long training and full data

| Run | Setup | Result |
|---|---|---|
| V2 | C48 recipe, 2 GPUs, 120 epochs on local train + val, monitored on local test | Local test mAP (v3) at epochs 70 / 80 / 90 / **100** / 110 / 120: 46.92 / 47.84 / 48.31 / **48.91** / 48.61 / 48.54; Codabench val 38.23 |
| L1 / L2 | V2 recipe, 100 epochs + D4 augmentation / + random zoom | Local test mAP 45.43 / 47.68, both below V2 |
| F1 | V2 recipe, all 9,392 images, 100 epochs | Codabench val 41.26 |
| F2 | F1 + A4 + A2, shift −0.2 | New leaderboard val **44.84** |

### 4.4 Post-processing (all on the outputs of the same forward pass)

| Step | Local effect | Notes |
|---|---|---|
| Detection-informed classification (λ = 1) | Weak models F1 about +1; strong models ±0.2 | λ chosen from {0, 0.5, 1}; the three are within noise on three models, so 1 stays |
| Box score × RPN objectness^0.25 + per-class soft-NMS (σ 0.75) | mAP +1.3 to +1.8 | σ 0.5–1 and power 0.25–0.5 all give about the same, so it is not tuned to the holdouts |
| Mean of the three Cascade stages' boxes | mAP +0.3 to +0.5 (3 models) | Matches how Cascade already averages the class probabilities; weighted versions differ by ≤0.06, so equal weights |
| Shift boxes by −0.4 px (−0.2 for A2 models) | mAP +0.4 to +0.7; A2 models +0.2 from a further −0.2 | A systematic annotation offset; see A2 in 5.3 |
| Score threshold 1e-5, at least 30 boxes per image | mAP +0.06 to +0.16 | 84 val images had no box at all; in COCO AP, low-score boxes at the end of the ranking only add recall |

Counter-examples: re-scoring boxes by the image-class probability changed nothing, so detection errors are mostly localisation, not class confusion. Re-scoring by the IoU-quality branch (B1) got worse the stronger it was (Δm −1.1 at strength 1).

### 4.5 AP by object size (A4's trade-off, local val + test, v3)

| Box side (512 input) | <8 | 8–16 | 16–32 | 32–96 | ≥96 |
|---|---|---|---|---|---|
| C36 | 8.2 | 17.0 | 28.4 | 52.4 | 47.6 |
| A4 | 12.9 | 20.7 | 28.2 | 50.4 | 47.6 |
| A4b | 12.5 | 21.3 | 29.6 | 49.8 | 47.8 |

## 5. How the method evolved: what was done at each step and why

### 5.1 Starting point

**Metric**: Δm = ½ × [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref] × 100. The reference mAP is low (26.13), so one mAP point is worth about 1.9 Δm and one F1 point only about 0.54, a ratio of 3.6. Classification is already near saturation at 93–96, so **the work concentrates on detection mAP**.

**Data**:
- 9 classes, 9,392 labelled SAR images.
- 70% of the boxes are COCO "small". At the 512 input, half the boxes are under 16 px, smaller than one ViT patch. Wind turbines, bridges and oil tanks have median sides of only 8–12 px.
- Bridge and playground AP started near 0. The labels are not wrong: a bridge is a 6–10 px dot where a river crosses a road, a playground a 15 px patch in dense city blocks.

**Rules** decide which routes are open: at most 130M parameters in total and 40M trainable (heads included); one model, no ensembles or TTA; each image once at ≤512, no tiling; no other SAR data or SAR-trained models; no training on val / test images.

### 5.2 The path at a glance

| Stage | Run | Change | Result | Decision |
|---|---|---|---|---|
| 0 | E00 | Reference: ViT-B/16 IN21k + LoRA r16 (attention only), 24 epochs | baseline | — |
| 1 Backbone | E01 | DINOv3 ViT-B/16 instead | local Δm +8.1 | ✅ |
| | E07 | DINOv3 ConvNeXt-T, all weights trained | +6.8 | ❌ |
| 2 Adaptation | E02 | IN21k + LoRA r32 on every linear layer + LayerNorms | +11.4 | stepping stone |
| | E09b | DINOv3 + LoRA r64 + extra settings | +6.2, detection worse | ❌ |
| | **E04r** | DINOv3 + the E02 recipe | **+18.1** | ✅ new base |
| | D6 | E04r, LoRA rank 64 | only +0.6 over E04r | ❌ |
| 3 Detection head | **D1** | + 3-stage Cascade R-CNN | +21.0 (mAP +1.3) | ✅ |
| | **D2** | + NWD matching + 6 px anchors | +19.7 (mAP +0.7) | ✅ |
| | D3 / D4 / D5 | Multi-level neck / D4 + copy-paste / Mona | +5.1 / +7.1 / crashed | ❌ |
| 4 Training length | L36 | E04r for 36 epochs | +29.4, the largest single gain | ✅ |
| | C36 / C48 | Cascade + NWD, 36 / 48 epochs | +31.3 / +38.6 | ✅ |
| | V2 | same, 2 GPUs, cosine 120 epochs | best at epoch 100 | ✅ 100 epochs |
| 5 Post-processing | — | soft-NMS, objectness, stage-box mean, shift, low-score boxes | about +3 mAP together | ✅ |
| 6 Full data | **F1** | V2 recipe, all 9,392 images, 100 epochs | **Codabench 41.26** | ✅ |
| 7 Refinement | A4 / A2 / AA | Size-aware IoU / flip-bias fix / both | see 4.2 | ✅ |
| | **F2** | F1 + A4 + A2, shift −0.2 at inference | **New leaderboard 44.84** | ✅ current best |

Local numbers are the mean of the two local holdouts, without post-processing.

### 5.3 Each stage: details and trade-offs

#### Stage 1: choosing the backbone

- **Tried**: DINOv3 ViT-B/16 (E01); DINOv3 ConvNeXt-T with full fine-tuning (E07); planned DINOv3 ConvNeXt-B (E03) and DINOv2 ViT-B/14.
- **Result**: swapping only the backbone for DINOv3 ViT-B/16 gave +8.1 Δm (+10.9 with post-processing), the largest single gain at the time.
- **ViT vs full fine-tuning of ConvNeXt-T**: ConvNeXt-T with its heads is only 35.5M, so it can be fully trained inside the 40M limit, and in the reference code full fine-tuning beats LoRA by about 13.5 Δm. In practice it gave +6.8: despite native stride-4 features, its AP below 8 px was only 1.9, worse than the reference. So DINOv3's weakness on small objects comes from its pretraining, not its architecture. Conclusion: a frozen 86M model with LoRA beats a fully trained 35M model.
- **DINOv2 ViT-B/14**: patch 14 doesn't divide 512, so the feature grid and the anchors would be misaligned; not used.
- **Cost**: DINOv3 is clearly stronger above 16 px and weaker than IN21k below it. The detection head and A4 later work on that gap.

#### Stage 2: adaptation (where LoRA goes, how wide)

- **Tried**: E02 (IN21k, LoRA on qkv / proj / fc1 / fc2, rank 32, trained LayerNorms, lr 3e-4); E09b (the "wide LoRA" from our literature search: DINOv3 + rank 64 + LoRA+ ×2 + drop path 0.1, trained biases and layer scales); E04r (DINOv3 + the E02 recipe unchanged); D6 (E04r with rank 64).
- **Result**: more adaptation capacity alone (E02, +11.4) gained more than the backbone alone (+8.1). E09b classified best (F1 95.2) but detected worse than E01. The diagnostic E04r showed that E09b's extra settings (drop path, LoRA+, stronger regularisation), not DINOv3, caused the drop. E04r reached +18.1, consistent on both holdouts, and became the base of every later run.
- **Rank 32 or 64**: r64 adds only +0.6, within noise, for 4.7M more trainable parameters. r32 keeps the budget for the detection head.
- **LayerNorms trained, patch embedding not**: LayerNorms are tiny and let the feature statistics adapt to SAR; training the patch embedding at a low lr (S6) gained very little.

#### Stage 3: the detection head and small objects

One change at a time on E04r, 24 epochs.

- **Cascade R-CNN (D1)**: 3 stages (IoU 0.5 / 0.6 / 0.7), light class-agnostic box heads, RoIAlign aligned to pixel centres. mAP +1.39 / +1.23 on val / test; the IN21k control V1 gained +1.6 too. The motive is localisation: AP50 and AP75 are far apart (aircraft 71 vs 16, wind turbines 46 vs 4), and an 8 px turbine at IoU 0.75 needs about 1 px precision. The light heads even cut about 1M trainable parameters (23.8M vs 24.9M).
- **NWD matching and small anchors (D2)**: the RPN matches by max(IoU, NWD) with C = 8, the smallest anchor is 6 px, anchors centred on pixels. An 8 px box a few pixels off has IoU near 0 and is dropped as a positive; NWD treats boxes as Gaussians and is far more forgiving for small boxes. Δm +1.6 (mAP +0.88 / +0.57).
- **Dropped**:
  - CNN spatial prior + multi-level neck (D3): much worse at 24 epochs, mainly because the multi-level neck takes the highest-resolution features from shallow blocks with weak semantics. The spatial prior alone (S5) gave +1.76 Δm at 36 epochs; that result came after F2 had started, so it heads the list for the next model.
  - D4 rotations + same-class copy-paste (D4): much worse; retested in a 100-epoch run (L1), still no help.
  - Mona adapters (D5): crashed with a cuDNN error at start; with surer options available it was not fixed.

#### Stage 4: training length (the largest single gain)

- Going from 24 to 36 epochs with the same recipe (L36) raised local Δm from +18.1 to +29.4; with Cascade + NWD, 36 epochs (C36) gave +31.3 and 48 epochs (C48) +38.6.
- V2: 2-GPU DDP, cosine 120 epochs, trained on local train + val and monitored on local test (curve in 4.3). It plateaued after epoch 90 with no clear overfitting.
- **100 epochs**: epochs 100 to 120 differ by about 0.4, within the noise of a 940-image set, and 100 is cheaper.
- **Cosine to 0, no EMA**: with the lr decayed to 0 the weights are already smooth; EMA (E1) came out even.
- **2-GPU DDP**: same global batch of 16, 37 → 72 images/s, 100 epochs in 3.6 h instead of 7, which made the full-data runs possible in time.

#### Stage 5: post-processing

Effects and reasons per step are in 4.4. Principles: everything works on one forward pass, no ensemble or TTA, and each step must point the same way on several models and both holdouts.

#### Stage 6: full data (F1)

- The V2 recipe on all 9,392 images for 100 epochs, no local holdout. Codabench val 41.26, 3.03 above V2 epoch 100 (38.23): mAP +1.35, F1 +0.84.
- **Trade-off**: with all data in training there is no holdout to monitor or pick checkpoints. That is acceptable because V2 showed epoch 100 is a plateau without overfitting, so the last epoch is taken; 940 more images (+11%) matter more than keeping the monitor.

#### Stage 7: refinement, F1 → F2

Both changes were screened on the 36-epoch C36 recipe; results in 4.2.

**A4: size-aware Cascade IoU thresholds** (`--cascade-size-iou`)

With the default 0.5 / 0.6 / 0.7, small objects get almost no positives in stages 2 and 3. A4 relaxes the thresholds for them:

| Box side | IoU thresholds of the three stages |
|---|---|
| < 8 px | 0.40 / 0.45 / 0.50 |
| 8–16 px | 0.45 / 0.50 / 0.55 |
| 16–32 px | 0.50 / 0.55 / 0.65 |
| larger | 0.5 / 0.6 / 0.7 (unchanged) |

- mAP +0.93 / +0.74 (against the single C36 run). By size (4.5), small objects gain a lot and 32–96 px loses 2 points.
- **Trade-off**: A4b tried to protect medium objects by relaxing only below 16 px. Its 32–96 px AP still fell by 2.6 and its total was about 0.35 below A4. So the medium-object loss doesn't come from their own thresholds but from the extra low-IoU positives of small objects acting on the shared classifier. A4 has the largest net gain and stays.

**A2: flip-bias correction** (`--flip-bias -0.3`)

- **Finding**: moving every predicted box by −0.4 px (512 input) raised mAP consistently on 3 models and both holdouts.
- **Ruling out other explanations**: shifting top-left and bottom-right corners separately, x and y separately, or in original-image pixels all still put the optimum at a uniform −0.4, with other variants within ±0.02. That is about −0.3 px in the original image, the same for 512 and 256 px originals, so a systematic offset in the annotation convention is the likely cause.
- **Consequence**: flip augmentation mirrors the offset, so the regression target of the same object swings between ±0.3 px, which matters for objects under 8 px.
- **Fix**: when flipping, move the box by twice the bias, so every training target follows one convention.
- **Result**: mAP +0.64 / +0.57, and the −0.4 shift is no longer needed, which confirms the explanation. A further −0.2 px shift adds +0.2 mAP on both holdouts for A2 models, while +0.2 hurts; F2 therefore uses −0.2.

**Do they stack (AA)?** +2.27 Δm over the C36 mean (mAP 38.43 / 40.01): they stack, but less than the sum of the two single gains.

**F2 on the leaderboard**:

| | Δm | macro-F1 | mAP | AP50 |
|---|---|---|---|---|
| F1 | 41.26 | 95.90 | 46.97 | 71.57 |
| F2 (shift −0.2) | **44.84** | 95.46 | **48.97** | 71.53 |

- mAP +2.00, more than the +1.0 the 36-epoch local runs suggested, perhaps because the leaderboard val set has more small-object classes (wind turbines).
- F1 −0.44 costs only about 0.24 Δm.
- AP50 is unchanged, so the whole gain is localisation at higher IoU thresholds, exactly what A4 and A2 target.

### 5.4 Tried and not adopted

| Run | Change | Result | Why |
|---|---|---|---|
| S8 | Overlapping patches (stride 8, 4× tokens) | mAP −0.13 / −0.64 | The smallest classes improved (bridge +0.7, playground +1.4, wind turbine +1.6) but oil tank fell from 42 to 35.6; 2.6× compute |
| E1 | Weight EMA | even | No extra smoothing once the cosine lr reaches 0 |
| B1 | IoU-quality branch | even; re-scoring worse | The stronger the re-scoring by predicted IoU, the worse |
| R1 | NWD constant 8 → 16 | unclear on test | within noise |
| R2 | 512 → 1024 sampled RoIs per image | about −2 mAP | clearly worse; not investigated further |
| S6 | Trained patch embedding | very small | not worth it |
| N1 | Last 2 blocks fully trained | small gain | 37.2M trainable, close to the 40M limit |
| L1 / L2 | 100 epochs + D4 / + random zoom | below V2 | neither augmentation helped |
| E07 | ConvNeXt-T full fine-tuning | +6.8 | see stage 1 |
| D3 / D4 | Multi-level neck / D4 + copy-paste | much worse | see stage 3 |

### 5.5 Principles throughout

1. **Read the metric before choosing a direction.** mAP moves Δm 3.6 times as much as F1, so most GPU time went to detection.
2. **Change one thing at a time, on the current best recipe.** E09b mixed several settings, and finding why detection dropped cost an extra diagnostic run (E04r); after that every screen changed one thing.
3. **Look at breakdowns, not just totals.** AP by size and by class showed that DINOv3 is weak on small objects, that localisation is the bottleneck, and that A4 costs medium objects.
4. **Rules first.** All post-processing uses one forward pass; the final model has 23.8M trainable and 109.0M total parameters, with room left under 40M / 130M.
5. **Improve both holdouts, then check on the leaderboard.** Every adopted change also paid off on the leaderboard: V2 38.23 → F1 41.26 → F2 44.84.

## 6. Efficiency

- 23.8M trainable parameters, 60% of the 40M limit and only 2.8M more than the reference (21.0M); 109.0M in total against the reference's 107.0M.
- The final model trains in about 7.2 GPU hours and predicts in 17.7 ms per image.
- Two costlier options were dropped because they didn't pay: S8 overlapping patches (2.6× compute, worse) and LoRA r64 (+4.7M trainable for +0.6).

## 7. Limitations and next steps

- The local holdouts are easier than the leaderboard (near-duplicates, different class mix); they rank methods, and the leaderboard gives the absolute score.
- Bridge and playground AP is still very low, 1–4 locally.
- **Next: F3 = F2 + S5 (spatial prior)**, possibly with N1. S5 gave +1.76 Δm locally (one seed) and works differently from A4 / A2, so the gains may stack; a full-data 2-GPU run takes about 3.6 h. S5's spatial prior is randomly initialised; with `--pretrained-prior` (ImageNet ResNet-18) it must be added to the pretrained-model table in 3.4.
- Phase 1 closes on 2026-11-01; Phase 2 (test, 2 submissions) opens on 11-02.
