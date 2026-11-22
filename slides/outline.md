# CS701 Talk — 10 min + 5 min Q&A (11 Nov 2026)

Target: ~9 slides, ~65 s per slide. Audience: class + TAs; they know the task,
so spend the time on *why the design looks like this* and *what the numbers say*.

## Slide 1 — Title (15 s)
- Project title, team number, members
- One line: "A frozen DINOv3 backbone + LoRA, with a detector rebuilt for
  sub-patch objects — validation Δm +38.2"
- One image: a SAR example with GT boxes (tiny objects circled)

## Slide 2 — Task in one slide (45 s)
- Input: one 8-bit gray SAR image (512² / 256²). Output: class (9) + boxes
- Metric: Δm = mean relative gain of macro-F1 and mAP vs ViT+LoRA reference
- Constraints: ≤130 M total / ≤40 M trainable, one model, no TTA, ≤512 px
- The catch: **50 % of boxes < 16 px = smaller than one ViT patch**

## Slide 3 — Why the obvious things fail (60 s)
- Reference LoRA: Δm 0 (F1 93.3 / mAP 26.1); from scratch: −20 → pretraining rules
- Full FT (over budget): +11.8 — the headroom is in *detection*, not classification
- For an 8 px box, a 2 px offset → IoU ≈ 0 → standard anchor matching starves
- SAR ≠ RGB: ImageNet stats, speckle, one channel

## Slide 4 — Our model (75 s) [the money slide: architecture diagram]
```
gray 512² → DINOv3 ViT-B/16 (FROZEN) + LoRA r32 all-linear + trained norms (23.8 M)
   mean token → LN+Linear → image class (9)
   stride-16 tokens → ViTDet pyramid (strides 4–64)
      → RPN (NWD matching, small cell-centred anchors)
      → 3-stage Cascade R-CNN (light class-agnostic heads)
post-proc: score = p_det × obj^0.25, soft-NMS σ.75, stage-mean boxes,
           −0.4 px bias correction; class = argmax log p_img + log q_det
```
- Everything shares the frozen backbone; heads = 20 M of the 23.8 M trainable

## Slide 5 — Experiments / numbers (75 s)
- Table: reference 0 → DINOv3 +8.1 → +wide LoRA +18.1 → +NWD&anchors/Cascade
  +21 → +both +31.3 → +long +38.6 → V2 leaderboard **+38.23**
- F1 95.06 / mAP 45.62 / AP50 70.38 on Codabench val
- Point out: mAP drives the score (+19.5 over ref); F1 +1.75

## Slide 6 — Post-processing gains (45 s)
- One table row each: soft-NMS + obj^0.25 (+1.3–1.8), stage-mean (+0.3–0.5),
  −0.4 px shift (+0.4–0.7), 30-box floor (+0.1), det-informed class (±0.2)
- Message: "free mAP on the same frozen outputs — still one forward pass"

## Slide 7 — What did NOT work (60 s) [bonus-friendly slide]
- token-stride 8 (2.6× compute, −0.4), LoRA r64, D4 + copy-paste,
  multi-level neck, ConvNeXt, Mona — all tried, all dropped
- cls-head on DINO-YOLO: +3–4 F1 but −1.5 mAP → kept detector vote
- Message: we screened on the local split, not on the leaderboard

## Slide 8 — Efficiency & the lighter family (45 s)
- Ours: 109 M / 23.8 M trainable, 15 ms/img, 3.5 h on 2×RTX 6000 Ada
- DINO-YOLO26-s: 31.9 M, single-shot, local Δm +21.3 — a third of the params
- Where the budget went: heads 20 M, LoRA ~3.5 M

## Slide 9 — Phase 2 plan + conclusion (45 s)
- F1_full: same recipe, all 9,392 labelled images, 100 epochs
- Two submissions only → F1_full + best variant; chose on local holdout,
  never on leaderboard feedback
- Take-away: freeze a strong generic backbone, spend the budget on wide LoRA
  + a tiny-object detector; detection is where Δm lives

## Backup slides (Q&A ammo)
- Per-class AP table (weak: bridge/playground/wind_turbine)
- Size-AP table (small/medium/large)
- The local-vs-leaderboard gap (48.9 vs 45.6 mAP)
- Submission format & validation (check_submission.py)
- GPU-hours bookkeeping for the report
