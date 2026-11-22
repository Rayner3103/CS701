# CS701 Team 7: multi-task learning on SAR images

**[Report (PDF)](report_latex/main.pdf)** · [Checkpoint release](https://github.com/Rayner3103/CS701/releases/tag/final-val-test-checkpoints-20261122) · [How to reproduce](#3-reproduce-the-final-results)

One model classifies a synthetic aperture radar (SAR) image into nine scene classes and detects its objects, within
the course limits (≤ 40 M trainable parameters, one forward pass at ≤ 512 px). Our final model is a frozen DINOv3
ViT-B/16 with rank-80 LoRA and a three-stage Cascade R-CNN rebuilt for tiny boxes, followed by zero-cost
post-processing on one set of averaged weights.

| Board | Entry | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|
| Phase 1, validation (1,426 images) | average of epochs 100–180, class-agnostic fusion | 95.27 | 55.10 | **56.49** |
| Phase 2, test | same weights, + min-keep 100 and lattice snapping | 96.96 | 55.74 | **61.25** |

- Report: [report_latex/main.pdf](report_latex/main.pdf) (CVPR format, 6 pages; source and build notes in
  [report_latex/](report_latex/README.md)).
- Final checkpoint: GitHub release
  [final-val-test-checkpoints-20261122](https://github.com/Rayner3103/CS701/releases/tag/final-val-test-checkpoints-20261122).
- The board scores above are inflated by train–test leakage that we measured after the labels were released
  (report Sec. 5): about a third of validation and test images are flipped or rotated pixel copies of a training image.

## Contents

**Report:** [report_latex/main.pdf](report_latex/main.pdf)

1. [Task and metric](#1-task-and-metric)
2. [Final model](#2-final-model)
3. [Reproduce the final results](#3-reproduce-the-final-results)
   - [3.1 Environment](#31-environment)
   - [3.2 Data](#32-data)
   - [3.3 Get the checkpoint](#33-get-the-checkpoint)
   - [3.4 Validation entry (Δm 56.49)](#34-validation-entry-δm-5649)
   - [3.5 Test entry (Δm 61.25)](#35-test-entry-δm-6125)
   - [3.6 Check the scores](#36-check-the-scores)
   - [3.7 Retrain from scratch](#37-retrain-from-scratch)
4. [Other results in the report](#4-other-results-in-the-report)
5. [Repository layout](#5-repository-layout)
6. [Course baseline and links](#6-course-baseline-and-links)

## 1. Task and metric

- Input: one grey SAR image. Output: its class, and a box, class and score for each object (all objects in an image
  share the image's class).
- 9 classes: aircraft, airport, bridge, car, harbor, oil tank, playground, ship, wind turbine. 9,392 labelled
  training images; 1,426 validation and the test images were scored on
  [Codabench](https://www.codabench.org/competitions/18271).
- Limits: one model, ≤ 130 M parameters in total and ≤ 40 M trainable, each image in once at ≤ 512 × 512 px, no
  ensembles or test-time augmentation, no SAR-pretrained weights.
- Ranking: classification macro-F1 and detection COCO mAP@[.5:.95], combined relative to the course reference model
  (ViT-B + LoRA + Faster R-CNN; validation 93.31 / 26.13, test 93.99 / 25.41):

  `Δm = 50 × [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref]`

## 2. Final model

| Part | Setting | Parameters |
|---|---|---:|
| Backbone | DINOv3 ViT-B/16 (timm `vit_base_patch16_dinov3.lvd1689m`), frozen; RGB patch kernel summed to one grey channel | 85.1 M frozen |
| Adaptation | LoRA rank 80, α 160 on `attn.qkv, attn.proj, mlp.fc1, mlp.fc2` of all 12 blocks; LayerNorms trained | 11.8 M |
| Neck | ViTDet pyramid P2–P6 (256 ch) + 64-channel spatial-prior CNN on raw pixels into strides 4 and 8 | 5.8 M |
| RPN | anchors 6–181 px, positives by max(IoU, NWD C=8) | 0.6 M |
| Detector | 3-stage Cascade R-CNN, width 1536, size-aware IoU thresholds, aligned RoIAlign | 21.7 M |
| Training | all 9,392 images, flips with −0.3 px flip-bias fix, AdamW (head 1e-4, LoRA 3e-4, wd 0.05), batch 16 on 2 GPUs, 200-epoch cosine, stopped after epoch 180 | |
| Weights | equal average of the checkpoints at epochs 100, 110, …, 180 | |
| Post-processing | 2,000 RPN proposals, no RPN objectness, mean of the 3 stages' boxes shifted −0.2 px, per-class soft-NMS σ 0.5, class-agnostic image–box score fusion, image label fused with detections (λ 2) | |
| **Total** | **125,076,960 parameters, 39,867,360 trainable** | |

Details and the reasons behind each part are in the report (Sec. 3) and in
[backups/final/materials/experiment_records/backup_56.49/README.md](backups/final/materials/experiment_records/backup_56.49/README.md).

## 3. Reproduce the final results

Two ways:

| Way | Needs | Time | Result |
|---|---|---|---|
| Inference with the released checkpoint (3.1–3.6) | 1 GPU (or `--device cpu`, slower), the val/test images | about 1 min per split on one RTX 6000 Ada | the submitted predictions, up to GPU float noise |
| Retrain from scratch (3.7) | 2 GPUs with ≥ 13 GB each, the training data | about 13 h, then inference | a new training run: scores vary by about ±0.5 Δm |

### 3.1 Environment

Run every step below from the repo root on `main`. Keep this folder layout, because the scripts look for the data
next to the repo:

```
CS701/
├── cs701-sar-benchmark/        # this repo
├── cs701-sar-course-data/      # raw download from Hugging Face
└── dataset/SARFact-Course-20K/ # converted training data (only needed to retrain)
```

```bash
mkdir CS701 && cd CS701
git clone https://github.com/Rayner3103/CS701.git cs701-sar-benchmark
cd cs701-sar-benchmark

conda create -n cs701bench python=3.12 -y && conda activate cs701bench
pip install -r requirements.txt   # torch 2.9.0 / torchvision 0.24.0 for CUDA 12.8, timm 1.0.24
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: 2.9.0+cu128 True
```

Exact package versions of the original run:
[backups/final/materials/experiment_records/backup_56.49/run/pip_freeze.txt](backups/final/materials/experiment_records/backup_56.49/run/pip_freeze.txt).

### 3.2 Data

From the top folder `CS701/`:

```bash
pip install -U huggingface_hub
hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir cs701-sar-course-data
cd cs701-sar-course-data && tar -xzf train.tar.gz && tar -xzf val.tar.gz && tar -xzf test.tar.gz && cd ..
```

This gives `cs701-sar-course-data/{train,val,test}/` (each with `images/` and `images.json`) and the released
labels `val_labels.csv` / `val_instances.json` used in 3.6. Only retraining (3.7) also needs the converted data:

```bash
cd cs701-sar-benchmark && python prepare_course_data.py   # writes ../dataset/SARFact-Course-20K/
```

More detail: [setup.md](setup.md) (Steps 3–4).

### 3.3 Get the checkpoint

The release holds two files (about 502 MB): `avg_e100-180.pt`, the 234 trained tensors averaged over epochs
100–180, and `model.safetensors`, the frozen DINOv3 backbone at the exact revision we used. From the repo root:

```bash
python backups/final/checkpoints/download.py           # downloads both assets into backups/final/checkpoints/ and checks SHA256
python backups/final/checkpoints/download.py --check   # later: verify the files without downloading
```

`backups/final/checkpoints/` then holds `avg_e100-180.pt`, `metrics.json` (the model configuration `predict.py`
rebuilds the model from) and the backbone in a Hugging Face cache layout. Point the Hugging Face cache at it so
nothing else is downloaded:

```bash
export HF_HUB_CACHE="$PWD/backups/final/checkpoints/hf_cache"
export HF_HUB_OFFLINE=1
```

### 3.4 Validation entry (Δm 56.49)

```bash
VAL=../cs701-sar-course-data/val
python predict.py --run backups/final/checkpoints --weights avg_e100-180.pt \
    --images $VAL --batch-size 16 --workers 2 \
    --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --stage-box mean \
    --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1 --topk 9 \
    --out submissions/val_avg100-180_agn.zip
python tools/check_submission.py submissions/val_avg100-180_agn.zip $VAL/images.json
```

`check_submission.py` checks the zip against the Codabench format rules (no labels needed).

### 3.5 Test entry (Δm 61.25)

The same weights and post-processing, plus `--min-keep 100` (keep at least 100 boxes per image) and snapping box
corners to the annotation lattice of the training labels:

```bash
TEST=../cs701-sar-course-data/test
python predict.py --run backups/final/checkpoints --weights avg_e100-180.pt \
    --images $TEST --batch-size 2 --workers 2 \
    --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --stage-box mean \
    --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1 --topk 9 --min-keep 100 \
    --out submissions/test_mk100.zip
python tools/snap_submission.py submissions/test_mk100.zip $TEST/images.json submissions/test_FINAL.zip
python tools/check_submission.py submissions/test_FINAL.zip $TEST/images.json
```

Caveat: the test zip was uploaded by hand and we cannot confirm whether the 61.25 entry was this FINAL candidate or
the plain one (the command above without `--min-keep 100` and snapping); both use the same checkpoint.

Predictions on another GPU or batch size can move a few low-scoring boxes (bf16 inference; on 1,426 validation
images we once saw 59,668 boxes instead of 59,711, with identical classification), so expect scores very close to
the board but not byte-identical zips. Compare the files inside the zips, not the zips' SHA256 (a zip
stores file times).

### 3.6 Check the scores

- **Official**: upload the zip to the [Codabench competition](https://www.codabench.org/competitions/18271)
  (validation or test phase).
- **Local, validation only**: the organisers released the validation labels after the competition. The leakage-audit
  scorer computes macro-F1, mAP and Δm on all validation images (reproduces the leaderboard to 0.01) and per
  leakage tier:

  ```bash
  python report_latex/leakage_analysis/score_bands.py \
      report_latex/leakage_analysis/results ../cs701-sar-course-data \
      submissions/val_avg100-180_agn.zip
  ```

  The second argument is the download folder holding `val_labels.csv` and `val_instances.json`. The printed `all:`
  group should read about `F1 95.27 mAP 55.10 dm 56.49`; `exact`, `near` and `far` are the leakage tiers (`far` is
  the report's "none" tier). It also writes `bands.json` into the results folder.

### 3.7 Retrain from scratch

From the repo root, with the converted data from 3.2. This is the command of
[train_BP7_cos200.sh](backups/final/materials/experiment_records/backup_56.49/train_BP7_cos200.sh), run from the
repo root:

```bash
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node 2 train.py \
  --data ../dataset/SARFact-Course-20K --out runs --name BP7_cos200 \
  --train-split train+val+test --eval-splits '' --seed 0 --batch-size 16 --workers 1 \
  --epochs 200 --schedule-epochs 200 --warmup-epochs 1 --stop-after-epoch 180 \
  --save-every 10 --save-training-state --reproducible-loader \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 80 --lora-alpha 160 --cascade-rep 1536 --spatial-prior 64 --train-norms \
  --lr 1e-4 --backbone-lr 3e-4 --weight-decay 0.05 \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" \
  --cascade-size-iou --flip-bias -0.3 --grad-ckpt --eval-every 10 --eval-batch-size 4
```

- `--train-split train+val+test` is all 9,392 labelled images (the local holdouts from `prepare_course_data.py`
  included); the Codabench val/test images are never trained on.
- The log should start with `parameters: 125,076,960 total, 39,867,360 trainable`.
- The run stops cleanly after epoch 180 while keeping the 200-epoch learning-rate schedule. Peak memory is 12.1 GB
  per GPU with `--grad-ckpt`; on free 48 GB GPUs drop it for speed (same maths). About 3 min 40 s per epoch on
  2 × RTX 6000 Ada.
- To resume after an interruption, add `--resume runs/BP7_cos200/training_state.pt` (model, optimizer, schedule and
  RNG are restored, so the trajectory is the same). `NCCL_P2P_DISABLE=1` worked around a PCIe peer-to-peer hang on
  our machine and is harmless elsewhere.

Average the nine checkpoints, then predict as in 3.4 and 3.5 with `--run runs/BP7_cos200 --weights avg_e100-180.pt`
(the run folder's `log.txt` holds the model configuration):

```bash
python backups/final/materials/experiment_records/backup_56.49/average_checkpoints.py \
    --dir runs/BP7_cos200 --out runs/BP7_cos200/avg_e100-180.pt
```

GPU training is not bit-deterministic (DDP, cuDNN), so a new run gives slightly different weights: about ±0.5 Δm on
the board for a single checkpoint, less after averaging.

## 4. Other results in the report

| Report item | Where |
|---|---|
| Table 1, progression of official scores | [backup_56.49/run/official_scores_phase1.json](backups/final/materials/experiment_records/backup_56.49/run/official_scores_phase1.json); earlier entries in [backups/](backups/) (folder names start with their val Δm) |
| Table 2, leave-one-out ablations (3 seeds) and Table 3, zero-training steps | [backups/final/materials/experiment_records/report_final_20261105/](backups/final/materials/experiment_records/report_final_20261105/) (`eval_tools/analyze_loo.py`, `eval_tools/ablate_pp.py`, logs) |
| Table 4 and Fig. 4, train–test leakage audit | [report_latex/leakage_analysis/](report_latex/leakage_analysis/), steps in [report_latex/README.md](report_latex/README.md) |
| Fig. 3a, CNN detector line (YOLO11-L, Faster/Cascade R-CNN on ConvNeXt-T/S) | developed by Le Ba Luat; its code and records are not in this repository (summary in report Secs. 3.2 and 4.3) |
| Fig. 3b and Sec. 4.3, full-block adaptation (F3) | 36-epoch screens: [backups/33.92-…](backups/33.92-8010-36ep-screens-B0-S1-S3-S4-S5-DINOv3-Cascade-NWD-SizeIoU-FlipBias-shift0.2/); 150-epoch F3 entry: [backups/52.56-F3-…](backups/52.56-F3-full150-avg120-140-DINOv3-LoRA48-Unfreeze2-Head1024-Cascade-NWD-SizeIoU-FlipBias-shift0.2/) |
| All experiment records, in English and Chinese | [backups/final/materials/README.md](backups/final/materials/README.md) |

## 5. Repository layout

```
train.py                 train + evaluate one configuration (all options: python train.py -h)
predict.py               predict a released split -> submission.zip, with the post-processing options above
pp_eval.py, pp_study.py  post-processing defaults and studies on raw model outputs
prepare_course_data.py   Hugging Face download -> ../dataset/SARFact-Course-20K/ with local holdouts
score_submission.py      score a zip on a labelled local holdout
make_submission.py       write_submission(): predictions -> Codabench zip
sarbench/
├── backbones.py         ViT-B/16, DINOv3, TerraMind backbones
├── adapters.py          LoRA, MoE-LoRA
├── model.py             backbone + class head + ViTDet pyramid + spatial prior + (Cascade) R-CNN
├── tiny.py              tiny-object parts: NWD matching, size-aware IoU, anchors
├── postprocess.py       score fusion, soft-NMS, stage averaging, min-keep
├── data.py, metrics.py  data loading and flips (with flip-bias fix); F1 and COCO mAP
└── training_state.py    exact resume (model, optimizer, schedule, RNG)
tools/                   checkpoint averaging, submission check, lattice snapping, experiment launchers
backups/                 experiment records of every main entry; backups/final has the final materials
report_latex/            the report (LaTeX, figures, leakage audit)
docs/                    course baseline README and design notes
```

## 6. Course baseline and links

- The course's original README (SAR basics, the TA reference model and its baseline results):
  [docs/BASELINE.md](docs/BASELINE.md); design notes: [docs/DETAILS.md](docs/DETAILS.md).
- Team workflow and data setup: [setup.md](setup.md). [USAGE.md](USAGE.md) describes an earlier candidate (F1_full),
  not the final model.
- Data: https://huggingface.co/datasets/doem1997/cs701-sar-course-data
- Leaderboard: https://www.codabench.org/competitions/18271
- Code: https://github.com/Rayner3103/CS701
