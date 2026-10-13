# CS701 SAR Multi-Task Benchmark

Reference code for the CS701 team assignment. One ViT-B/16 backbone, two tasks on SAR images: **classify** the
image (9 classes) and **detect** its objects.

![One SAR image, one shared backbone, two heads: a class and the boxes](docs/task_pipeline.png)

## Contents

1. [SAR Task Introduction](#1-sar-task-introduction)
   - [1.1 SAR Basics](#11-sar-basics)
   - [1.2 Task](#12-task)
   - [1.3 Metric](#13-metric)
   - [1.4 Small Objects](#14-small-objects)
2. [Quick Start](#2-quick-start)
   - [2.1 Install](#21-install)
   - [2.2 Prepare Course Data](#22-prepare-course-data)
   - [2.3 Train](#23-train)
   - [2.4 Training Options](#24-training-options)
   - [2.5 Make Submission](#25-make-submission)
3. [Baseline Results (for reference)](#3-baseline-results-for-reference)
4. [Code Structure](#4-code-structure)
5. [Links](#5-links)

## 1. SAR Task Introduction

### 1.1 SAR Basics

![A radar sends microwave pulses sideways, records the echoes, and combines them along its flight path into one long virtual antenna](docs/sar_explainer.gif)

- **Active radar**: satellite or aircraft sends microwave pulses, records echoes. Works day, night, through clouds.
  Looks sideways.
- **Synthetic aperture**: echoes along the flight path combined into one long virtual antenna, so fine resolution.
- **Pixel = echo strength**: 8-bit grey, no colour. Metal and corners bright, calm water dark, grainy speckle.

<details>
<summary>What SAR images look like</summary>

![Speckle, bright metal and corners, dark smooth surfaces, bright and dark slopes](docs/sar_effects.png)

</details>

### 1.2 Task

- Input: one SAR image. Output: its class + a box, class and score for each object.
- Each image: one class label, 1 or more boxes, all of that class.
- 9 classes: aircraft, airport, bridge, car, harbor, oil tank, playground, ship, wind turbine.
- Data, formats, release dates: [dataset card](https://huggingface.co/datasets/doem1997/cs701-sar-course-data).
  Rules: [Codabench](https://www.codabench.org/competitions/18271).
- Limits: one model; at most 130 M parameters in total (frozen ones count; this code: 106-107 M) and at most 40 M
  trainable (heads included: LoRA 21.0 M; full fine-tuning and from scratch, 106 M, are over); each image in once at
  ≤ 512 × 512 px; no ensembles, no test-time augmentation. Your report states parameters (total, trainable) and
  compute; full list on Codabench (Terms).

### 1.3 Metric

- Classification: macro-F1. Detection: COCO mAP.
- Ranking: **Δm**, mean relative change of macro-F1 and mAP vs a reference model (ViT + LoRA, the example command
  in 2.3; mean of 3 seeds):
  `Δm = 100% × ½ [(F1 − F1_ref) / F1_ref + (mAP − mAP_ref) / mAP_ref]`
- Δm = 0: as good as the reference. Δm > 0: better. ViT full fine-tuning: about +13 (over the trainable budget: an
  upper bound).

### 1.4 Small Objects

Half of all boxes are smaller than one 16 × 16 ViT patch (at 512 × 512 input).

![A close-up with the 16 px patch grid: wind turbines of 6-10 px and the 15.9 px median box are smaller than one patch](docs/patch_grid.png)

## 2. Quick Start

### 2.1 Install

```bash
conda create -n cs701bench python=3.12 -y && conda activate cs701bench
pip install -r requirements.txt   # CUDA 12.8 builds of torch 2.9.0 / torchvision 0.24.0
```

### 2.2 Prepare Course Data

```bash
python prepare_course_data.py   # -> ../dataset/SARFact-Course-20K, the layout train.py reads
```

- Converts the [course data](https://huggingface.co/datasets/doem1997/cs701-sar-course-data) download and holds out
  10% + 10% of train (stratified, `--seed 0`) as local val and test: the released val/test have no labels.
- Images are symlinked, not copied. Full walkthrough (download, layout, smoke test): [docs/setup.md](docs/setup.md).

### 2.3 Train

One run, on a 10 GB GPU (e.g. RTX 3080), with early stopping:

```bash
mkdir -p runs/single
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python train.py --backbone terramind --init pretrained \
    --adapt moelora --batch-size 8 --epochs 50 --eval-every 2 --patience 5 --out runs/single \
    >> runs/single/run.log 2>&1 &
```

- TerraMind + MoE-LoRA had the best test Δm in our own runs (`runs/full_seed0`: +5.0 on the local split, seed 0).
  Add `--spatial-prior` to try the small-object branch.
- Results go to `runs/single/terramind_pretrained_moelora/`. Files in it: [docs/usage.md](docs/usage.md#outputs).
- `PYTORCH_CUDA_ALLOC_CONF` reduces memory fragmentation: without it, TerraMind + MoE-LoRA runs out of memory at batch
  size 8 on 10 GB.
- It runs in the background and keeps going after you close the terminal. Follow it with
  `tail -f runs/single/terramind_pretrained_moelora/log.txt`. `runs/single/run.log` also catches crash messages.
- 50 epochs take about 5.5 h on an RTX 3080 (24 epochs took 2.7 h), less if early stopping triggers.
- Interrupted? Run the same command with `--resume` (keep `--epochs 50`).

Other runs:

```bash
python train.py --backbone vit --init pretrained --adapt lora   # the Δm reference; one run, about 50 min
bash run_all.sh                                                  # the eight reference runs, about 7 h
bash run_all_3080.sh 24                                          # the four adapter runs, without and with the
                                                                 #   spatial prior, at batch size 8: docs/usage.md
```

- `--adapt full` and `--init scratch` train 106 M parameters: over the 40 M budget, for reference only.
- Pretrained weights download from Hugging Face on first use (ViT 0.4 GB, TerraMind 1.5 GB).
- Times for one RTX PRO 6000 Blackwell GPU. Memory: [docs/DETAILS.md](docs/DETAILS.md#time-and-memory).

### 2.4 Training Options

All options of `train.py` (`python train.py -h`):

| Option | Default | What it does |
|---|---|---|
| `--backbone vit\|terramind` | required | ViT-B/16 (ImageNet-21k) or TerraMind-1.0-base (Earth observation) |
| `--init pretrained\|scratch` | required | `scratch` only with `--adapt full` |
| `--adapt full\|lora\|moelora` | required | fine-tune the whole backbone, or freeze it and train LoRA / MoE-LoRA adapters |
| `--spatial-prior` | off | a small CNN on the raw image feeds the neck at strides 4 and 8 (small objects); the run folder gets `_sp` |
| `--epochs` | 24 | one epoch of warmup, then cosine decay to 0 at the last epoch |
| `--batch-size` | 16 | set for a 96 GB GPU; 8 on 10-12 GB, 4 if `CUDA out of memory` |
| `--lr` | 1e-4 | new modules: heads, neck, adapters |
| `--backbone-lr` | `--lr` | trainable backbone parameters; `run_all.sh` uses 2e-5 for full fine-tuning |
| `--weight-decay` | 0.05 | AdamW weight decay |
| `--eval-every` | 4 | validate every N epochs and after the last one; the best val Δm so far is saved as `best_model.pt` |
| `--patience` | off | early stopping: stop once N validations in a row have not beaten the best val Δm (see below) |
| `--resume` | off | continue from the run folder's `last_checkpoint.pt`; needs the same `--epochs` |
| `--seed` | 0 | random seed |
| `--out` | `runs` | the run folder is `OUT/<backbone>_<init>_<adapt>[_sp]` |
| `--data` | `../dataset/SARFact-Course-20K` | the prepared data (2.2) |
| `--workers` | 8 | data-loading processes |
| `--limit` | all | only the first N images of every split (quick checks) |

Early stopping (`--patience`):

- It counts validations, not epochs: `--eval-every 2 --patience 3` stops after 6 epochs without improvement.
- The epoch where training stops becomes the "last" model (`model.pt`, `test (last, epoch k)` in `log.txt`).
  `metrics.json` records `last_epoch` and `stopped_early`.
- The learning-rate schedule still spans all `--epochs`, so a run that stops early never decays to 0. Give it the
  full `--epochs` you would otherwise train.
- Works with `--resume`, and `run_all_3080.sh` counts a run that stopped early as complete.

### 2.5 Make Submission

```bash
python predict.py --run runs/single/terramind_pretrained_moelora --images ../dataset-v2/val   # -> submission.zip
```

- Predicts on the released, unlabelled images (`--images`: the folder holding `images.json`) with the run's
  `best_model.pt` (`--weights model.pt` for the last epoch), and writes `submission.zip` for Codabench.
- It calls `write_submission` from `make_submission.py` (the function in the briefing video), which maps boxes back to
  original image pixels.
- Compare several runs first: `python compare_runs.py runs/full_seed0`.

## 3. Baseline Results (for reference)

Test split, %, mean ± std over 3 seeds. Δm reference: ViT + LoRA, mean of its 3 seeds (93.99 / 25.41), so its Δm
is 0. Full fine-tuning and from scratch train 106 M parameters with the heads: over the 40 M budget, shown for
reference (full fine-tuning: the upper bound).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 95.8 ± 0.3 | 31.8 ± 0.2 | +13.5 ± 0.4 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 94.0 ± 0.2 | 25.4 ± 0.5 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 93.4 ± 0.5 | 26.8 ± 0.3 | +2.5 ± 0.8 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 95.6 ± 0.2 | 30.8 ± 0.4 | +11.5 ± 0.7 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.6 ± 0.4 | 26.5 ± 0.4 | +1.4 ± 1.0 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 92.4 ± 0.1 | 27.8 ± 0.3 | +3.8 ± 0.7 |
| ViT, from scratch | full training | 86.0 M | 82.8 ± 0.7 | 17.0 ± 0.3 | −22.5 ± 0.9 |
| TerraMind, from scratch | full training | 85.3 M | 86.1 ± 0.2 | 17.8 ± 0.5 | −19.3 ± 0.8 |

<details>
<summary>Val results</summary>

Val split, %, mean ± std over 3 seeds. Δm reference: the same configuration on val (93.31 / 26.13).

| backbone | adaptation | trained backbone params | macro-F1 | mAP | Δm |
|---|---|---:|---:|---:|---:|
| ViT (ImageNet-21k) | full fine-tuning | 86.0 M | 94.4 ± 0.3 | 32.0 ± 0.4 | +11.8 ± 0.7 |
| ViT (ImageNet-21k) | LoRA | 0.88 M | 93.3 ± 0.4 | 26.1 ± 0.4 | 0.0 ± 0.9 |
| ViT (ImageNet-21k) | MoE-LoRA | 1.03 M | 92.2 ± 0.8 | 27.1 ± 0.2 | +1.3 ± 0.9 |
| TerraMind (Earth observation) | full fine-tuning | 85.3 M | 94.2 ± 0.3 | 31.5 ± 0.1 | +10.7 ± 0.3 |
| TerraMind (Earth observation) | LoRA | 0.88 M | 92.2 ± 0.4 | 27.0 ± 0.5 | +1.0 ± 1.3 |
| TerraMind (Earth observation) | MoE-LoRA | 1.03 M | 91.9 ± 0.4 | 28.8 ± 0.1 | +4.4 ± 0.3 |
| ViT, from scratch | full training | 86.0 M | 83.5 ± 0.9 | 18.2 ± 0.5 | −20.3 ± 1.3 |
| TerraMind, from scratch | full training | 85.3 M | 86.0 ± 0.9 | 19.2 ± 0.4 | −17.3 ± 1.2 |

</details>

Accuracy and AP50: [docs/DETAILS.md](docs/DETAILS.md#results-in-full).

## 4. Code Structure

```
prepare_course_data.py  course data download -> the layout train.py reads, with local val/test hold-outs
train.py              entry: train + evaluate one configuration
run_all.sh            the eight reference runs (calls train.py)
run_all_3080.sh       the four adapter runs, without and with --spatial-prior, on a 10 GB GPU (docs/usage.md)
compare_runs.py       one line per run: test macro-F1, mAP, Δm, best epoch
predict.py            a trained run on unlabelled images -> submission.zip
visualize.py          browse a run's predictions vs ground truth (docs/visualize_usage.md)
make_submission.py    write_submission(): predictions -> submission.zip
sarbench/             the code behind train.py
├── data.py           images + boxes at 512 × 512, flips; boxes back to original pixels
├── backbones.py      ViT (ImageNet-21k), TerraMind-1.0-base
├── adapters.py       LoRA, MoE-LoRA on a frozen backbone
├── model.py          backbone + class head + ViTDet feature pyramid + Faster R-CNN
└── metrics.py        accuracy, macro-F1, COCO box AP
tests/                checks of sarbench/ (python -m pytest tests)
docs/setup.md         from nothing to a smoke test: clone, environment, data
docs/usage.md         run_all_3080.sh: commands, resuming, outputs, troubleshooting
docs/DETAILS.md       model, training protocol, design notes, time and memory
docs/visualize_usage.md  visualize.py: commands, options, keys
docs/improvement.md   ideas for improving the model and research directions
```

More: [docs/DETAILS.md](docs/DETAILS.md).

## 5. Links

- Data: https://huggingface.co/datasets/doem1997/cs701-sar-course-data
- Leaderboard: https://www.codabench.org/competitions/18271
- Questions: TA Zichen, zichen.tian.2023@phdcs.smu.edu.sg
