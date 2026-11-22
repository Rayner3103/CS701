# USAGE: the current best model (F1_full)

How the model behind `F1_full_val.zip` (Junkai's best Phase 1 submission candidate) is built, how to
reproduce it, and how to test a change against it. Setup, data download and the local split are in
[setup.md](setup.md) (Steps 1-4); every command below runs from the repo folder.

## 1. Results

| Model | Training data | Codabench val Δm | macro-F1 | mAP | AP50 |
|---|---|---|---|---|---|
| V2, epoch 100 (`V2_e100_val.zip`) | local train+val (8,452 images) | **38.23** | 95.06 | 45.62 | 70.38 |
| F1_full (`F1_full_val.zip`, this document) | all 9,392 labelled images | not yet submitted | | | |

F1_full is the V2 recipe with two changes: it trains on all labelled images (the local test split
included, no holdout left) for 100 epochs, the epoch where V2 peaked on its holdout, and it uses the
final post-processing of section 2.5. The local holdout over-estimates Codabench: V2 epoch 100 scored
mAP 48.91 / F1 95.59 on the local test split but 45.62 / 95.06 on Codabench val.

## 2. Method

One model, one forward pass per image, both tasks:

```
gray image, stretched to 512 x 512
  -> DINOv3 ViT-B/16 (frozen) + LoRA          32 x 32 tokens of 768
  -> mean of the tokens -> LayerNorm + Linear  image class (9)
  -> ViTDet simple feature pyramid             strides 4, 8, 16, 32, 64
  -> RPN (NWD matching) -> 3-stage Cascade R-CNN -> boxes, classes, scores
```

### 2.1 Backbone
- `vit_base_patch16_dinov3.lvd1689m` from timm (Meta DINOv3, pretrained on 1.7 B generic web images, no SAR).
  timm sums the RGB patch-embedding weights into one gray channel.
- Frozen. Trained: LoRA rank 32, alpha 64 on `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` of all 12 blocks,
  plus every LayerNorm (`--train-norms`).

### 2.2 Detection head (`sarbench/tiny.py`)
- **Cascade R-CNN**, 3 stages (IoU 0.5 / 0.6 / 0.7), light class-agnostic box heads, aligned RoIAlign.
  Scores are the mean of the 3 stages' class probabilities (standard Cascade inference).
- **RPN matching by NWD** (normalized Gaussian Wasserstein distance, C = 8) instead of IoU alone: IoU is
  near zero for an 8 px box offset by a few pixels, NWD is not.
- **Anchors** 6, 8, 11 / 16, 19, 23 / 32, 38, 45 / 64, 76, 91 / 128, 152, 181 px on the five levels,
  centred on the pixel grid (`--anchor-offset 0.5`). Half of all boxes are smaller than one 16 px patch.

### 2.3 Training
- Batch 16, AdamW (weight decay 0.05), LoRA lr 3e-4, heads lr 1e-4, one warmup epoch, cosine to 0 at
  epoch 100, bf16, `torch.compile`, flips only (no other augmentation).
- 2 GPUs with DDP (each GPU takes 8 images per step; the global batch stays 16).

### 2.4 Size and compute (what the report must state)
- Parameters: **109,014,752 total, 23,805,152 trainable** (limits 130 M / 40 M); `train.py` prints both.
- Training: 2 x NVIDIA RTX 6000 Ada (48 GB), 3.5 h for F1_full, peak 38 GB per GPU.
- Inference: about 15 ms per image on one GPU (bf16), soft-NMS on the CPU.
- Input: each image once at 512 x 512 (stretched if not square; no tiling, no upscaling beyond 512),
  normalised with the train-set mean / std (0.2526 / 0.2133).

### 2.5 Post-processing (`pp_eval.FINAL`, the default of `predict.py`)
All of it works on the outputs of the same single forward pass (no ensemble, no test-time augmentation):

| Step | Local effect |
|---|---|
| Box score = detector probability x RPN objectness^0.25, then per-class Gaussian soft-NMS (sigma 0.75) | about +1.3 to +1.8 mAP |
| Box = mean of the three Cascade stages' boxes | +0.3 to +0.5 mAP (3 models) |
| Boxes moved by -0.4 px (a systematic annotation offset measured on the local holdouts) | +0.4 to +0.7 mAP |
| Score threshold 1e-5, at least 30 boxes per image, at most 100 | +0.06 to +0.16 mAP |
| Image class = argmax of log p_classifier + log (highest detector probability of that class) | about ±0.2 F1 |

### 2.6 What each part added (local holdouts, Δm vs our run of the reference model, no post-processing)

| Run | Change | Epochs | Local Δm |
|---|---|---|---|
| E00 | reference: ViT-B/16 IN-21k + LoRA r16 | 24 | 0 |
| E01 | DINOv3 backbone | 24 | +8.1 |
| E04r | + LoRA r32 on every linear layer, trained norms | 24 | +18.1 |
| D1 / D2 | + Cascade R-CNN / + NWD matching and small anchors | 24 | +21.0 / +19.7 |
| C36 | + both | 36 | +31.3 |
| C48 | same, longer | 48 | +38.6 |
| V2 | same, 120 epochs on train+val (local test only) | 100 / 120 | best at epoch 100 |

Tried and not used: overlapping stride-8 patches (`--token-stride 8`, 2.6x compute, -0.4 mAP: better on
bridge / playground / wind turbine, much worse on oil tank), LoRA rank 64, D4 rotations + copy-paste, a
multi-level ViT neck, ConvNeXt backbones, Mona adapters.

## 3. Reproduce F1_full

```bash
conda activate cs701bench
# 2 GPUs, about 3.5 h. NCCL_P2P_DISABLE=1: PCIe peer-to-peer hangs NCCL on our machine (harmless elsewhere).
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node 2 train.py \
  --name F1_full --train-split train+val+test --eval-splits '' --save-every 10 --epochs 100 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 32 --lora-alpha 64 --train-norms --backbone-lr 3e-4 --compile \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181"
```

- One GPU: the same arguments with `python train.py` instead of `torchrun ...` (about 7 h, peak about 43 GB).
- Keep the quotes around `--anchor-sizes`: the `;` would end the shell command.
- The run goes to `runs/F1_full/`: `model.pt` (the trained weights only, about 100 MB; the frozen DINOv3
  weights are reloaded from timm), `metrics.json` (arguments, parameter counts, time, peak memory),
  `log.txt`, and `ckpt_e10.pt` ... `ckpt_e90.pt`.

## 4. Make and check a submission

```bash
# Phase 1 (val); for Phase 2 use ../cs701-sar-course-data/test
python predict.py --run runs/F1_full --images ../cs701-sar-course-data/val --out submission_val.zip \
  --raw-out runs/F1_full/raw_codabench_val.pt
python tools/check_submission.py submission_val.zip ../cs701-sar-course-data/val/images.json
```

- `predict.py` uses the final post-processing by default; `--weights ckpt_e90.pt` picks a checkpoint,
  `--device cpu` runs without a GPU (about 0.7 s per image).
- `--raw-out` keeps every image's raw outputs. Re-scoring them with other post-processing then takes
  seconds and no model: add `--from-raw runs/F1_full/raw_codabench_val.pt` to the same command.
- `tools/check_submission.py` checks the Evaluation-page format rules (files, header, one row per image,
  labels 0..8, category_id 1..9, finite boxes with w, h > 0) and reports boxes outside their image and
  whether the box classes agree with the image class.

## 5. Test a change against the current recipe

Screen on the local split (train on local train, score local val and test), then compare with C36:

```bash
python train.py --name my_change --epochs 36 --dump-raw \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 32 --lora-alpha 64 --train-norms --backbone-lr 3e-4 --compile \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" \
  <your change>
python pp_eval.py --run runs/my_change      # final post-processing -> runs/my_change/metrics_pp.json
python tools/size_ap.py runs/my_change      # AP by box size and per class
```

C36 under the final post-processing (the bar to beat): local val mAP 37.22 / macro-F1 95.54, local test
mAP 39.04 / macro-F1 95.16 (about 2 h on one GPU). One run of a configuration varies by about ±1 Δm.

Useful `train.py` options added on this branch: `--train-split` / `--eval-splits` / `--eval-every` /
`--save-every` (growth curves and checkpoints), `--ema`, `--aug d4`, `--zoom-p`, `--cascade-size-iou`,
`--iou-branch`, `--flip-bias`, `--token-stride`, `--unfreeze-last`, `--spatial-prior`; run
`python train.py --help` for the full list.
