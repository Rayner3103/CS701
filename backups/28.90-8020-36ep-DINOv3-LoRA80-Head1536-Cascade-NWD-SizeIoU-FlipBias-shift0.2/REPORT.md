# Report material: the scaling experiment SC1 (local 80:20)

**English** · [中文](REPORT.zh-CN.md)

> **Purpose: material for the CS701 team report**, as the ablation "scale the model up to the parameter budget". Reproduction steps are in [README.md](README.md); the report material for F2 itself is in `../44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/REPORT.md`.
>
> Numbers checked on 2026-10-27 against `run/metrics_pp.json`, `run/size_ap_vs_AA.json` and the training logs.

---

## 0. Before writing the report

1. **Local Δm 43.75 and leaderboard Δm 28.90 are different things.** Local Δm is relative to our local reference run E00. On the leaderboard SC1 scores 28.90, far below F2's 44.84, because of 80% of the data and 36 epochs, not because scaling failed (see 2.5). The real leaderboard test of scaling is the full-data SC2.
2. **One seed.** SC1 ran once. The +5.92 Δm gain is far beyond the C36 seed noise (±0.70), but the report should still say "one seed"; add 1–2 seeds if time allows.
3. **Total parameters stop short of 130M.** All new parameters are trainable, so trainable hits the 40M limit while the total is 124.9M. Going higher would mean adding frozen parameters, which would not help.
4. **Gradient checkpointing changes only speed and memory.** It was on because GPU0 is shared; batch, learning rate and results are unchanged, so SC1 and AA compare directly.
5. **The report must state 39.7M trainable.** The Terms reward fewer trainable parameters at a similar Δm; SC1 spends the whole budget to buy score. If SC2 beats F2 on the leaderboard only slightly, weigh whether to switch.

## 1. Why scale, and how

F2 trains 23.8M parameters, 60% of the 40M limit, with 109.0M in total (limit 130M). The aim here: **keep the architecture, widen the existing modules in proportion until the budget is used**, and see how much that buys.

The size knobs in the current architecture are the LoRA rank (`--lora-rank`), the Cascade head width (`--cascade-rep`), and fully training the last blocks (`--unfreeze-last`). The last only turns frozen weights trainable, adds no parameters and is not a widening, so it was not used.

Candidate configurations (models built on CPU and counted):

| LoRA rank | Head width | Trainable | Total | Left under 40M |
|---|---|---|---|---|
| 32 (F2) | 1024 (F2) | 23.81M | 109.01M | 16.19M |
| 64 | 1024 | 28.52M | 113.73M | 11.48M |
| 64 | 1536 | 37.30M | 122.51M | 2.70M |
| 96 | 1280 | 37.43M | 122.64M | 2.57M |
| 72 | 1536 | 38.48M | 123.69M | 1.52M |
| **80** | **1536** | **39.66M** | **124.87M** | **0.34M** |

The choice is **LoRA rank 80 (α 160) + head width 1536**: 99.1% of the trainable budget, with the extra 15.9M split between backbone adaptation (+7.1M) and the detection heads (+8.8M), so neither side alone is widened. In the earlier 24-epoch runs, raising only the rank to 64 (D6) gained very little (+0.6, within noise), which is why the heads were widened too.

## 2. Results

### 2.1 Overall (final post-processing, no shift)

| Model | val F1 / mAP / AP50 | test F1 / mAP / AP50 | Local Δm |
|---|---|---|---|
| C36 (37 runs) | 95.39 / 37.46 / — | 95.21 / 38.91 / — | 35.56 ± 0.70 |
| AA (F2's recipe, 36 epochs) | 95.05 / 38.43 / 65.85 | 95.55 / 40.01 / 68.27 | 37.83 |
| **SC1** | **95.65 / 40.96 / 67.96** | 95.05 / **42.87** / **70.25** | **43.75** |
| SC1 − AA | +0.60 / +2.53 / +2.11 | −0.50 / +2.86 / +1.98 | **+5.92** |

- Both holdouts improve: Δm val +5.97, test +5.87.
- Almost all of the gain is detection: mAP +2.5 / +2.9. Classification F1 goes up on one holdout and down on the other, within noise.
- With a further −0.2 px shift: mAP 41.21 / 43.17 (+0.25 / +0.30); the full-data SC2 submission also uses −0.2.

### 2.2 By object size (mean of local val and test)

| Box side (512 input) | all | <8 | 8–16 | 16–32 | 32–96 | ≥96 |
|---|---|---|---|---|---|---|
| AA | 39.2 | 14.8 | 19.7 | 28.4 | 50.0 | 46.0 |
| SC1 | 41.9 | 20.4 | 26.0 | 31.8 | 53.2 | 46.0 |
| Change | +2.7 | **+5.6** | **+6.3** | +3.4 | +3.2 | 0.0 |

**Small objects gain the most.** The two bins below 16 px rise by 5.6 and 6.3 points; ≥96 px is unchanged. Unlike A4, which helped small objects at the cost of 2 points on 32–96 px, SC1 also gains 3.2 on medium objects. Wider heads and more LoRA capacity help small-object localisation most.

### 2.3 By class (mean AP over local val and test)

| Class | AA | SC1 | Change |
|---|---|---|---|
| oil_tank | 45.8 | 54.1 | **+8.3** |
| harbor | 50.5 | 55.2 | +4.7 |
| wind_turbine | 43.1 | 47.0 | +3.9 |
| airport | 59.1 | 62.8 | +3.7 |
| playground | 3.7 | 5.5 | +1.8 |
| aircraft | 49.8 | 51.4 | +1.6 |
| bridge | 1.7 | 2.9 | +1.2 |
| car | 59.1 | 59.5 | +0.4 |
| ship | 40.2 | 39.0 | **−1.2** |

- Oil tanks gain most (+8.3): more than half their boxes are under 16 px, often in dense clusters.
- Ship is the only class that drops (−1.2), within the noise of a single class's AP.
- Bridge and playground stay very low; scaling does not fix them.

### 2.4 During training (evaluations every 4 epochs, no post-processing)

val / test mAP:

| Epoch | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|
| AA | 15.5 / 15.3 | 20.9 / 21.2 | 25.6 / 26.0 | 27.4 / 27.7 | 31.3 / 33.0 | 33.7 / 35.4 | 36.3 / 38.2 | 37.0 / 38.7 |
| SC1 | 14.3 / 14.3 | 20.4 / 20.1 | 24.8 / 25.9 | 28.7 / 30.3 | 32.8 / 35.2 | 36.3 / 38.3 | 38.7 / 40.5 | 39.8 / 41.6 |

SC1 trails slightly for the first 12 epochs (more parameters, slower start), overtakes AA from epoch 16, and the gap keeps widening to about 3 points at epoch 32. The larger model needs enough epochs; the 100-epoch full-data run should let it use them.

### 2.5 Leaderboard (Phase 1 val, not posted)

| Submission | Training data | Epochs | Δm | macro-F1 | mAP | AP50 |
|---|---|---|---|---|---|---|
| SC1 | 80% | 36 | 28.90 | 94.69 | 40.85 | 67.64 |
| V2, epoch 100 | 90% | 100 | 38.23 | 95.06 | 45.62 | 70.38 |
| F2 | 100% | 100 | 44.84 | 95.46 | 48.97 | 71.53 |

- **28.90 is far below F2, but that was expected and does not show that scaling failed.** The difference is mainly training data and epochs, not parameter count.
- One mAP point is worth about 1.9 Δm; SC1 is 8 mAP below F2, about 16 Δm.
- The same recipe went from 41.5 to 48.9 local test mAP between 48 and 100 epochs; SC1 trained 36.
- Local scores overestimate the leaderboard: SC1 local test mAP 42.87 → leaderboard 40.85 (about 2 lower); V2 was about 3 lower.
- The fair comparison is AA, also 36 epochs on 80% of the data. AA was never submitted; with the same gap of about 2 points it would score about 38 mAP and Δm about 24, so SC1 is about 5 higher, in line with the local +5.92. **This is an estimate, not a measurement.**

## 3. What the Terms ask the report to state (SC1)

| Item | Value |
|---|---|
| Parameters | 124,866,272 total, 39,656,672 trainable (plus 16 floating-point buffer values); breakdown in README section 2 |
| GPU | 1 × NVIDIA RTX 6000 Ada (48 GB), shared with other users |
| Training time | 2.64 h (36 epochs on local train, 7,512 images) |
| Peak memory | 12.3 GB (gradient checkpointing) |
| Input and preprocessing | As F2: 512×512 gray, each image once |
| Pretrained models | As F2: DINOv3 ViT-B/16 LVD-1689M |
| Seeds | 1 (seed 0) |
| Inference time | Not measured yet; wider heads, so probably a little slower than F2's 17.7 ms per image |

## 4. Conclusion

- On the same split and epoch budget, using the full parameter budget raises local Δm from 37.83 to 43.75 (+5.92), consistent on both holdouts: the largest gain of any single change so far.
- The gain is concentrated on small objects (+6 AP below 16 px) and localisation.
- Cost: trainable parameters go from 23.8M to 39.7M, and training and inference are slower. The report's efficiency item should say so.

## 5. Next steps

- **SC2 (full data, 150 epochs) has finished**: the same configuration on all 9,392 images for 150 epochs, 2-GPU DDP, started 2026-10-28 02:07, 6.38 h, peak 10.6 GB per GPU, inference 16.1 ms per image. Run folder `cs701_runs/SC2_full150/`; submissions `submissions/SC2_e{150,140,130,120}_val_shift-0.2.zip`, all format-checked. Submit e150 first, then e130 if quota allows; if one beats F2 (44.84), post the best. (Single-GPU 100- and 150-epoch attempts were stopped at epochs 8 and 22 in favour of two GPUs.)
- Add 1–2 seeds so the +5.92 can be reported as mean ± std.
- Add SC2's leaderboard score here and give SC2 its own backup folder.
