# F2_full (shift −0.2): backup and reproduction

**English** · [中文](README.zh-CN.md)

`submission/F2_full_val_shift-0.2.zip` (12,363,762 bytes) is the best Phase 1 submission of the Junkai branch so far: val Δm **44.84** on the new leaderboard. This folder holds everything needed to reproduce it: code, commands, arguments, logs, package versions and the submission file itself.

This README says what is in the folder and how to reproduce the result. The numbers for the report, the method's evolution and its trade-offs are in [REPORT.md](REPORT.md).

## 1. Results

| Submission | Contents | Platform | Δm | macro-F1 | mAP | AP50 |
|---|---|---|---|---|---|---|
| `F1_full_val.zip` | v2 recipe, 100 epochs on all 9,392 labelled images | Codabench val (retired) | 41.26 | 95.90 | 46.97 | 71.57 |
| **`F2_full_val_shift-0.2.zip`** (this backup) | F1 + A4 + A2, every box moved by −0.2 px at inference | New leaderboard, Phase 1 val | **44.84** | 95.46 | 48.97 | 71.53 |

- F2 beats F1 by 3.58 Δm: mAP +2.00, macro-F1 −0.44.
- Local ablation: 37 runs of the C36 recipe (36 epochs) give val / test mAP 37.46±0.35 / 38.91±0.36. With A4 and A2 both on, the same setup gives 38.43 / 40.01. The −0.2 shift adds 0.2 mAP on each of the two local holdouts.

## 2. Method

- **Backbone**: DINOv3 ViT-B/16 (timm `vit_base_patch16_dinov3.lvd1689m`, pretrained on generic web images, no SAR data). The backbone is frozen; only LoRA (rank 32, alpha 64, on qkv / proj / fc1 / fc2 of every block) and the LayerNorms are trained.
- **Detection**: 3-stage Cascade R-CNN. The RPN matches anchors by NWD (C = 8), the smallest anchor is 6 px, and RoIAlign is aligned to pixel centres.
- **A4** (`--cascade-size-iou`, table `a4`): the IoU thresholds of the three cascade stages depend on object size; small objects get lower thresholds.
- **A2** (`--flip-bias -0.3`): the annotated boxes carry a small systematic offset, and flip augmentation mirrors it to the other side. A2 moves flipped boxes back by the measured bias, so all training targets follow one coordinate convention.
- **Training**: all 9,392 labelled images, 100 epochs, cosine learning rate, bf16, `torch.compile`, 2-GPU DDP, global batch 16.
- **Post-processing** (the `predict.py` defaults): box score × RPN objectness^0.25, per-class soft-NMS; boxes are the mean of the three stages' boxes; score threshold 1e-5 with at least 30 boxes per image; the image class uses the detections too. On top of that, `--shift -0.2` moves every box by −0.2 px.
- **Size and rules**: 109,014,752 parameters in total, 23,805,152 trainable (limits 130M / 40M). One model, no TTA, each image goes in once at 512×512.

## 3. What is in the folder

| Path | Contents | In git? |
|---|---|---|
| `code/` | Code snapshot taken at training time (`cs701_runs/F2_full/code`), identical file by file to commit `51257a2` | yes |
| `train_F2_full.sh` | Retrains F2 | yes |
| `make_submission_F2.sh` | Writes and checks the shift −0.2 submission | yes |
| `run/args.json`, `run/metrics.json` | All train.py arguments, parameter counts, training time, peak memory | yes |
| `run/log.txt` | Full training log | yes |
| `run/a3a1_F2_full_ddp.sh`, `run/F2_extra_args.txt` | The original queue script that ran (paths are this machine's) | yes |
| `run/pip_freeze.txt` | Package versions of the `cs701bench` conda environment | yes |
| `submission/` | The submission file and its SHA256 | yes |
| `weights/model.pt` | The trained weights (trainable part only, 95 MB) | no, local only |
| `weights/F2_full_val_raw.pt` | Raw model outputs on the Phase 1 val images (338 MB) | no, local only |
| `README*.md`, `REPORT*.md` | This guide and the report material, each in English and Chinese | yes |

The weights are not in git: GitHub caps single files at 100 MB, and once committed every clone downloads them. On Junkai's machine they are at `/mnt/hdd/junkaichen/CS701/CS701/backups/44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/weights/`; the original run folder `/mnt/hdd/junkaichen/CS701/cs701_runs/F2_full/` has another copy plus `ckpt_e10` to `ckpt_e90`.

## 4. Reproduce

Run every command below from this folder.

### 4.0 Three ways to reproduce

| Way | Needs | Time | Result |
|---|---|---|---|
| A. Re-score only | local `weights/F2_full_val_raw.pt`, no GPU | seconds | both files in the zip byte-identical to the original |
| B. Predict with the trained weights | local `weights/model.pt`, 1 GPU | about 1 min | same classes, same box count, box scores within 0.003 |
| C. Train from scratch | 2 GPUs with about 48 GB | about 3.6 h + prediction | equal within seed noise (see 4.6) |

`weights/` exists only on Junkai's machine; elsewhere only way C is possible.

### 4.1 Environment

```bash
conda create -n cs701bench python=3.12 -y
conda activate cs701bench
pip install -r code/requirements.txt   # torch 2.9.0+cu128, torchvision 0.24.0, timm 1.0.24, pycocotools 2.0.11
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.9.0+cu128 True
```

- The full package list of the original environment is in `run/pip_freeze.txt`.
- Original hardware: 2 × NVIDIA RTX 6000 Ada (48 GB), CUDA 12.8.
- The first run downloads the DINOv3 backbone `timm/vit_base_patch16_dinov3.lvd1689m` (about 0.35 GB) from Hugging Face. `model.pt` stores only the trained weights and the frozen backbone is reloaded every time, so prediction also needs Hugging Face access or a copy cached in `HF_HOME`.

### 4.2 Data

Follow Steps 3–4 of the repo's `setup.md`:

```bash
hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir cs701-sar-course-data
cd cs701-sar-course-data && tar -xzf train.tar.gz && tar -xzf val.tar.gz && cd ..
cd <repo> && python prepare_course_data.py   # writes ../dataset/SARFact-Course-20K
```

- `prepare_course_data.py` uses a fixed seed (0) to split the 9,392 labelled images into local train / val / test of 7,512 / 940 / 940.
- F2 uses `--train-split train+val+test`, so all three parts are trained on. The official val / test images have no labels and are never trained on.
- Both scripts look for the data in the `setup.md` layout. If yours is elsewhere, set:

| Variable | Meaning | Value on Junkai's machine |
|---|---|---|
| `DATA` | the converted data | `/mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K` |
| `IMAGES` | the leaderboard images (val or test) | `/mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/val` |

### 4.3 Train (way C)

```bash
DATA=/path/to/dataset/SARFact-Course-20K ./train_F2_full.sh
```

The script runs the original command (see `run/a3a1_F2_full_ddp.sh`) with the absolute paths replaced by variables:

```bash
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node 2 train.py \
  --data "$DATA" --out runs --name F2_full \
  --train-split train+val+test --eval-splits '' --save-every 10 --epochs 100 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 32 \
  --lora-alpha 64 --train-norms --backbone-lr 3e-4 --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3
```

| Flag | What it does |
|---|---|
| `--train-split train+val+test --eval-splits ''` | Train on all labelled data; no local evaluation split |
| `--epochs 100` | One warmup epoch, then cosine learning rate down to 0 |
| `--backbone dinov3 --init pretrained --adapt lora ...` | Frozen DINOv3 ViT-B/16 with LoRA r32 / α64 on four linear layers of every block |
| `--train-norms` | Also train every LayerNorm |
| `--backbone-lr 3e-4` | LoRA learning rate; the heads use the default `--lr 1e-4` |
| `--cascade --aligned-roi` | 3-stage Cascade R-CNN, RoIAlign aligned to pixel centres |
| `--rpn-nwd 8 --anchor-offset 0.5 --anchor-sizes=...` | NWD anchor matching (C = 8), 3 sizes per level from 6 px, anchors centred on pixels |
| `--cascade-size-iou` | A4: IoU thresholds relaxed by object size (default table `a4`) |
| `--flip-bias -0.3` | A2: correct the annotation offset under flip augmentation |
| `--compile` | `torch.compile`, about 1.3× faster |
| `NCCL_P2P_DISABLE=1` | PCIe P2P hangs NCCL on this machine; harmless elsewhere |

Everything else is a default; the full list is in `run/args.json`: batch 16 (8 per GPU), AdamW, weight decay 0.05, bf16, flips only, seed 0.

- **Output** goes to `runs/F2_full/`: `model.pt` (95 MB), `metrics.json`, `log.txt`, `ckpt_e10.pt` … `ckpt_e90.pt`.
- **Checks**: the log should start with `parameters: 109,014,752 total, 23,805,152 trainable`. The original run took 3.59 h, peaked at 38.0 GB per GPU and ran at about 36 images/s; the loss over the last epochs sits between 0.05 and 0.3, which you can compare with `run/log.txt`.
- **With one GPU**: replace `torchrun --standalone --nproc_per_node 2` with `python` and keep every flag; about 7 h, peak about 43 GB. A smaller `--batch-size` also runs, but then the result differs from the original run.

### 4.4 Make the submission

```bash
# Ways B / C: predict with the weights (weights/ if present, else runs/F2_full)
IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_F2.sh

# Way A: no GPU, re-score the saved raw outputs
FROM_RAW=1 IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_F2.sh

# Phase 2: test images (raw outputs exist only for val, so ways B / C only)
IMAGES=/path/to/cs701-sar-course-data/test ./make_submission_F2.sh
```

The script runs `predict.py --shift -0.2` and then `tools/check_submission.py`. The output is `F2_full_<val|test>_shift-0.2.zip`; set `OUT=...` to change it.

The default post-processing of `predict.py` is the final one (`pp_eval.FINAL`):
1. Box score = detector probability × RPN objectness^0.25, then per-class Gaussian soft-NMS (σ = 0.75).
2. Boxes are the mean of the three Cascade stages' boxes.
3. Score threshold 1e-5, at least 30 and at most 100 boxes per image.
4. Image class = argmax(log p_classifier + log highest detector probability of that class).
5. Shift: models trained with `--flip-bias` are not shifted by default; here `--shift -0.2` is set explicitly (in 512-input pixels).

A passing check prints:

```
OK: 1426 images, 79969 boxes, 0 images without boxes, at most 100 boxes per image and class ..., 0 boxes outside their image, 99.9% of boxes scored > 0.3 have category_id = label + 1
```

### 4.5 Check the result

```bash
sha256sum -c submission/SHA256SUMS   # the original submission file
python - <<'EOF'
import zipfile
a = zipfile.ZipFile('F2_full_val_shift-0.2.zip'); b = zipfile.ZipFile('submission/F2_full_val_shift-0.2.zip')
for n in ('classification.csv', 'detection.json'):
    print(n, a.read(n) == b.read(n))
EOF
```

Don't compare the SHA256 of two zips: a zip records file times, so the hash differs even when the contents match.

Checked on 2026-10-27:
- `code/` matches commit `51257a2` file by file.
- Way A: both files `True`.
- Way B: `classification.csv` is `True`; `detection.json` has the same 79,969 boxes with scores at most 0.0024 apart, which is bf16 GPU arithmetic and negligible for mAP.
- Way C was not run.

### 4.6 How closely retraining reproduces it

The seed is fixed at 0, but DDP, `torch.compile` and cuDNN benchmark are not bit-deterministic on GPU, so retrained weights will not be identical.

The size of that noise was measured on C36: 37 runs give local val / test mAP 37.46±0.35 / 38.91±0.36. A 100-epoch full-data run should vary by about as much, roughly ±0.5 Δm on the leaderboard.

### 4.7 Reproduce the local ablation: AA (A4 + A2) against C36

F2's two changes over F1 were tested in 36-epoch local runs: train on local train (7,512 images), evaluate on local val / test (940 each).

```bash
cd code
python train.py --data "$DATA" --out ../runs --name AA_a4a2 --epochs 36 --dump-raw \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 32 --lora-alpha 64 --train-norms --backbone-lr 3e-4 --compile \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" \
  --cascade-size-iou --flip-bias -0.3
python pp_eval.py --run ../runs/AA_a4a2      # final post-processing -> ../runs/AA_a4a2/metrics_pp.json
```

Drop `--cascade-size-iou --flip-bias -0.3` from the last line for the C36 baseline. About 2 h on one GPU.

| Run | Local val mAP | Local test mAP |
|---|---|---|
| AA | 38.43 | 40.01 |
| C36 (37 runs) | 37.46±0.35 | 38.91±0.36 |

### 4.8 Troubleshooting

- **`torchrun` hangs at the first step**: set `NCCL_P2P_DISABLE=1` (the script does).
- **Out of GPU memory**: batch 16 on one GPU needs about 43 GB. Use two GPUs, or a smaller `--batch-size` (changes the result).
- **`predict.py` fails to load with `_orig_mod`**: old code; the version in `code/` handles weights saved by `--compile` runs.
- **Another checkpoint**: add `--weights ckpt_e90.pt` to `predict.py`.
- **No GPU**: `predict.py --device cpu`, about 0.7 s per image.
