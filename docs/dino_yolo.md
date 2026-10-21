# DINO-YOLO

Back to the [README](../README.md).

A second model family next to the reference ViT + Faster R-CNN: YOLO12-s whose backbone also gets the features of a
DINOv3 ViT-S/16 run on the same image, trained with the ultralytics trainer. Code: `sarbench/dino_yolo.py`,
`train_yolo.py`, `eval_yolo.py`, `predict_yolo.py`, `prepare_yolo_data.py`.

## Model

```
image [B,3,512,512] (gray in three channels)
  YOLO12-s backbone (COCO-pretrained)  -> P3/8, P4/16, P5/32
  DINOv3 ViT-S/16 (timm, LVD-1689M)    -> last block [B,384,32,32]
      resized to P3, P4, P5 (bilinear), 1x1 conv each, added to the backbone output   (--fuse 4 6 8)
  YOLO12 neck + Detect head (9 classes)
image class = the class with the highest total detection score
```

- The design follows Sompote/DINOV3-YOLOV12 (DINOv3 features fused into the YOLO12 backbone), written as one
  module on the official `ultralytics` package. Unlike that repo, it never falls back silently to DINOv2 or random
  weights.
- The 1x1 fusion convs start at zero, so training starts from exactly the COCO-pretrained YOLO12-s
  (`tests/test_dino_yolo.py`).
- **No classification head by default.** Every box of an image has the image's class, so the image class is read
  from the detections. `--cls-head` adds one instead (see Next steps). An image without detections gets class 0 (aircraft). The prediction threshold is 0.001, so this
  hardly ever happens.
- **Parameters: 31.24 M in total, all trainable**: DINOv3 ViT-S/16 21.59 M, YOLO12-s and the fusion convs 9.65 M.
  That is within the limits of 130 M total and 40 M trainable. `DinoYOLOTrainer.get_model` checks the limits and
  stops if they are exceeded.
- The DINO ViT trains at 0.1x the learning rate (`--dino-lr-mult`), so the pretrained features change gently.

## Training protocol

- AdamW, lr 1e-3, weight decay 0.05, 3 warmup epochs, cosine decay to 1e-5, batch size 16, AMP (fp16).
- 512 px input (letterbox). Ultralytics augmentation: mosaic (off for the last 10 epochs), scale 0.5, translate 0.1,
  horizontal and vertical flips, brightness. No hue or saturation, because the images are gray. No test-time
  augmentation.
- 90 / 180 / 270° rotations (`--rot90`, sarbench/dino_yolo.py `RandomRot90`): after the flips, with probability 0.75 a
  uniformly chosen quarter, half or three-quarter turn, so all 8 orientations of an image are equally likely. Off by
  default (`--rot90 0`): at 24 epochs `--rot90 0.75` lowered test mAP by 3.1 / 2.7 and Δm by 6.4 / 5.2 for DINO +
  YOLO12-s / YOLO26-s (`*_rot90_ep24` runs). It is not saved in the checkpoint, so give it again with `--resume`.
- best.pt is chosen on local val by the ultralytics fitness (its mAP50-95). Early stopping comes after 20 epochs
  without improvement (`--patience`).
- The data is the same local split as train.py (`prepare_course_data.py`, seed 0: 7,512 / 940 / 940).

## Commands

```bash
conda create -n cs701yolo --clone cs701bench && conda activate cs701yolo
pip install -r requirements-yolo.txt          # adds ultralytics 8.4.170

python prepare_yolo_data.py                   # -> ../dataset/SARFact-YOLO: symlinked images, YOLO labels, data.yaml
python -m pytest tests/test_dino_yolo.py      # CPU, no downloads

# Smoke run: 5% of train for 2 epochs, then scoring (about 3 min)
python train_yolo.py --fraction 0.05 --epochs 2 --out runs/yolo_smoke --name smoke

# One run in the background, about 2.5 min per epoch on an RTX 3080 (7.5 GB at batch size 16)
mkdir -p runs/yolo
PYTORCH_ALLOC_CONF=expandable_segments:True nohup python train_yolo.py --epochs 24 --name dino_yolo12s_ep24 \
    >> runs/yolo/dino_yolo12s_ep24.log 2>&1 &
python train_yolo.py --name dino_yolo12s_ep24 --resume          # after an interruption

python eval_yolo.py --run runs/yolo/dino_yolo12s_ep24           # rescore: metrics.json, predictions_{val,test}.json
python compare_runs.py runs/yolo                                # same columns as the reference runs
python predict_yolo.py --run runs/yolo/dino_yolo12s_ep24 --images ../dataset-v2/val   # -> submission.zip
```

- `train_yolo.py` scores best.pt on local val and test when training ends. It uses the metrics of train.py
  (macro-F1, COCO mAP via pycocotools in original pixels, Δm), so the results are comparable with the reference
  table.
- Final submission: train with `--data ../dataset/SARFact-YOLO/data_full.yaml` (all of train), using the epoch count
  chosen on the local split. Its val scores are then on training images and only show that training works.
- `yolo12s.pt`, `yolo26s.pt` (COCO weights) and `weights/yolo26n.pt` (the ultralytics AMP check) are downloaded into
  the working folder and ignored by git.

## YOLO26

- `--yolo yolo26s` swaps YOLO12-s for YOLO26-s (9.95 M: NMS-free end-to-end head, no DFL). DINO + YOLO26-s is 31.9 M
  in total, all trainable. The DINO features go to layers 4, 6 and 10, the backbone outputs the YOLO26 head reads
  (`neck_inputs`; YOLO12: 4, 6, 8).
- `--yolo yolo26s-p2` adds a P2/4 output for the 5-10 px objects (DINO also goes to layer 2). It starts from
  `yolo26s.pt` (`--pretrained`), but only 6.1 of its 9.7 M YOLO parameters load: the backbone and the top-down neck.
  The P2 path and the Detect head start random.
- `--dino none` trains the plain YOLO with the same protocol: the baseline for the DINO fusion.
- The protocol is the YOLO12 one (AdamW, not YOLO26's default MuSGD), so the rows compare. The end-to-end head gives
  at most 100 boxes per image without NMS (`iou` has no effect).
- A frozen DINO (and DINO + YOLO26-m, 43.5 M, under the 40 M trainable limit): see Frozen DINO + LoRA.

```bash
python train_yolo.py --yolo yolo26s --dino none --epochs 24 --name yolo26s_ep24      # plain YOLO26-s
python train_yolo.py --yolo yolo26s --epochs 24 --name dino_yolo26s_ep24             # DINO + YOLO26-s
python train_yolo.py --yolo yolo26s-p2 --epochs 24 --name dino_yolo26s-p2_ep24       # DINO + YOLO26-s-p2
```

## Frozen DINO + LoRA

`--dino-adapt frozen | lora` freezes the DINO ViT, so a bigger one fits: frozen weights count towards the 130 M total,
not towards the 40 M trainable. DINOv3 ViT-B/16 (`--dino vit_base_patch16_dinov3.lvd1689m`, 85.6 M) is the biggest
that fits next to YOLO26-s; ViT-L/16 (303 M) does not.

| `--dino-adapt` (ViT-B/16 + YOLO26-s) | total | trainable | peak memory, batch 16 | time per step (RTX 3080) |
|---|---:|---:|---:|---:|
| `full` with ViT-S/16 (as above) | 31.9 M | 31.9 M | 6.1 GB | 211 ms |
| `frozen` | 96.4 M | 10.7 M | 3.1 GB | 188 ms |
| `lora` (qkv, proj; rank 16) | 97.3 M | 11.6 M | 7.8 GB (8.7 GB in training) | 372 ms |
| `lora --dino-grad-ckpt` | 97.3 M | 11.6 M | 3.7 GB | 472 ms |
| `lora --dino-grad-ckpt --lora-targets attn.qkv attn.proj mlp.fc1 mlp.fc2` | 98.7 M | 13.1 M | 3.8 GB | 532 ms |

Forward and backward of the model alone, fp16; the training run adds the EMA copy and the data loader.

- `frozen`: the ViT runs without gradients and keeps no activations, so it is cheaper than the fine-tuned ViT-S.
  The YOLO and the 1x1 fusion convs train.
- `lora`: LoRA (sarbench/adapters.py, alpha = 2 x rank as train.py) on `--lora-targets` of every block (default
  `attn.qkv attn.proj`, the reference's 0.88 M; `mlp.fc1 mlp.fc2` add 1.47 M at rank 16), `--lora-rank` 16. The LoRA
  weights are in the DINO learning-rate group: lr x `--dino-lr-mult` = 1e-4, the reference LoRA rate. B starts at 0,
  so training starts from the frozen ViT. Without `--dino-grad-ckpt` a 10 GB GPU is nearly full at batch 16.
- Use the `lvd1689m` ViTs without `_qkvb`: their attention calls `qkv` as a layer, which LoRA wraps; the `_qkvb` ones
  apply its weight directly.
- The ultralytics trainer turns gradients on for every parameter before it builds the optimizer.
  `DinoYOLOTrainer.build_optimizer` freezes the ViT again (`DinoYOLO.freeze_dino`), leaves its frozen weights out of
  the optimizer, and checks the limits again (`tests/test_dino_yolo.py`).
- The settings are in the model yaml, so `--resume`, eval_yolo.py and predict_yolo.py rebuild the same model.

```bash
B="--yolo yolo26s --dino vit_base_patch16_dinov3.lvd1689m --epochs 24"
python train_yolo.py $B --dino-adapt frozen --name dinob_frozen_yolo26s_ep24
python train_yolo.py $B --dino-adapt lora --dino-grad-ckpt --name dinob_lora_yolo26s_ep24
python train_yolo.py $B --dino-adapt lora --dino-grad-ckpt --lora-targets attn.qkv attn.proj mlp.fc1 mlp.fc2 \
    --name dinob_loramlp_yolo26s_ep24
```

## Results

Local split, seed 0, %. Δm as in train.py and compare_runs.py (reference 93.31 / 26.13), so the rows are comparable
with `runs/`. With the test reference of the leaderboard (93.99 / 25.41), dino_yolo12s_ep24 is at about +8.8.

| run | epochs | params (total / trainable) | val F1 | val mAP | test F1 | test mAP | test Δm |
|---|---:|---:|---:|---:|---:|---:|---:|
| dino_yolo12s_ep24 (best.pt = epoch 24) | 24 | 31.2 M / 31.2 M | 91.1 | 30.7 | 90.4 | 30.8 | +7.4 |
| dino_yolo12s_cls_ep24 (`--cls-head`, best.pt = epoch 24) | 24 | 31.2 M / 31.2 M | 94.1 | 28.9 | 94.5 | 29.4 | +6.9 |
| yolo26s_ep24 (plain YOLO26-s, `--dino none`, best.pt = epoch 24) | 24 | 9.95 M / 9.95 M | 86.6 | 28.2 | 88.2 | 28.3 | +1.4 |
| dino_yolo26s_ep24 (best.pt = epoch 24) | 24 | 31.9 M / 31.9 M | 91.7 | 31.4 | 90.6 | 31.8 | +9.4 |
| dino_yolo26s-p2_ep24 (best.pt = epoch 24) | 24 | 31.7 M / 31.7 M | 91.8 | 30.3 | 91.4 | 30.6 | +7.6 |
| dinob_frozen_yolo26s_ep24 (frozen ViT-B/16, `--dino-adapt frozen`, best.pt = epoch 24) | 24 | 96.4 M / 10.7 M | 90.3 | 28.4 | 88.6 | 28.5 | +2.1 |
| dinob_lora_yolo26s_ep24 (ViT-B/16 + LoRA r16 qkv, proj, best.pt = epoch 24) | 24 | 97.3 M / 11.6 M | 91.0 | 31.1 | 90.3 | 31.5 | +8.7 |
| dinob_loramlp_yolo26s_ep24 (ViT-B/16 + LoRA r16 qkv, proj, fc1, fc2, best.pt = epoch 24) | 24 | 98.7 M / 13.1 M | 91.4 | 31.1 | 90.7 | 31.2 | +8.3 |
| dino_yolo12s_rot90_ep24 (`--rot90 0.75`, best.pt = epoch 24) | 24 | 31.2 M / 31.2 M | 90.0 | 28.0 | 89.5 | 27.8 | +1.1 |
| dino_yolo26s_rot90_ep24 (`--rot90 0.75`, best.pt = epoch 24) | 24 | 31.9 M / 31.9 M | 90.4 | 29.2 | 90.7 | 29.1 | +4.3 |
| dino_yolo12s_ep50 (best.pt = epoch 49) | 50 | 31.2 M / 31.2 M | 92.2 | 33.9 | 92.2 | 34.3 | +15.0 |
| dino_yolo12s_ep100 (best.pt = epoch 100) | 100 | 31.2 M / 31.2 M | 92.9 | 37.2 | 92.4 | 36.3 | +19.0 |
| dino_yolo26s_ep100 (best.pt = epoch 100) | 100 | 31.9 M / 31.9 M | 93.7 | 37.2 | 93.9 | 37.1 | +21.3 |
| terramind + MoE-LoRA (`runs/full_seed0`) | 24 | | | | 90.9 | 29.4 | +5.0 |
| vit + MoE-LoRA + spatial prior (`runs/full_seed0`) | 24 | | | | 92.8 | 28.6 | +4.5 |
| terramind + MoE-LoRA (`runs/single_seed0`) | 50 | | 90.2 | 33.6 | 91.8 | 35.2 | +16.5 |
| terramind + MoE-LoRA (`runs/single_seed0_ep100`, best epoch 86) | 100 | | 90.2 | 36.8 | 91.8 | 38.0 | +21.8 |

- 24 epochs took 59 min on an RTX 3080. Val mAP was still rising at the last epoch: longer runs should gain more, as
  the TerraMind runs did from 24 to 50 epochs.
- 50 epochs (about 2 h) beat TerraMind + MoE-LoRA at 50 epochs on val (Δm +14.3 vs +12.7), with higher macro-F1
  (92.2 vs 90.2). It ran all 50 epochs, so a 100-epoch run followed.
- 100 epochs (about 4 h): val Δm +20.9 vs +18.8 for TerraMind + MoE-LoRA at 100 epochs, but on test it is behind
  (+19.0 vs +21.8). TerraMind's test mAP is 1.7 points higher. The best val epoch was the last one (fitness 37.5), so
  training had not yet plateaued. DINO-YOLO keeps the higher macro-F1 on both splits.
- Detection carries the score. mAP 30.8 is close to the full fine-tuning upper bound of the reference (31.8), but
  macro-F1 is about 2.4 points below the adapter runs.
- YOLO26 at 24 epochs: DINO + YOLO26-s beats DINO + YOLO12-s on val and test (val Δm +9.2 vs +7.6, test mAP 31.8 vs
  30.8, at the same F1). The DINO features add 3.2 / 3.6 val / test mAP over the plain YOLO26-s (+1.4). The P2 variant
  is behind DINO + YOLO26-s in mAP (−1.1 / −1.2): its P2 path and Detect head start random, and it takes 1.5 h vs
  0.9 h. Its wind turbine AP (18.8 vs 18.0) and F1 are a little higher. All three still improved at epoch 24.
- 90 / 180 / 270° rotations (`--rot90 0.75`, 24 epochs) hurt both models: test mAP −3.1 / −2.7 and Δm −6.4 / −5.2
  for DINO + YOLO12-s / YOLO26-s (val: −2.8 / −2.2 mAP). The loss is in detection: aircraft, airport, harbor and
  wind turbine AP fall by 4-7 points. Test F1 changes by −0.9 / +0.1. The rotated runs trail by 2-3 val mAP from
  epoch 10 on and the gap is not closing at epoch 24, so slower convergence alone does not explain it. A likely cause
  (not tested): SAR is side-looking, so radar shadow and layover fall along a consistent image direction, which a
  quarter turn moves to one the test images never show; flips keep it on the same axis.
  Bridge and playground do not improve: bridge test AP 3.6 → 3.3 / 3.5 → 3.5, playground 1.2 → 0.9 / 2.2 → 1.2;
  bridge test F1 −3.9 / −0.8, playground +0.2 / −0.6 (val F1 −5.2 / −1.9 and −3.0 / −4.3). Playground val images
  called bridge rise from 16 to 25 / 15 to 22. Rotation is therefore off by default (`--rot90 0`).
- DINO + YOLO26-s at 100 epochs (3.9 h): test Δm +21.3 vs +19.0 for DINO + YOLO12-s at 100 epochs (F1 +1.5, mAP
  +0.8), and level with TerraMind + MoE-LoRA at 100 epochs (+21.8; one seed each, so within noise). On val it is the
  best run (Δm +21.3 vs +20.9 and +18.8). It has the highest F1 of the 100-epoch runs (93.9); TerraMind keeps 0.9
  more test mAP. With the leaderboard test reference it is at about +23.0. Val mAP was flat over the last 5 epochs
  (37.8-37.9 fitness), so longer training would gain little.
- Frozen DINOv3 ViT-B/16 + YOLO26-s at 24 epochs (Frozen DINO + LoRA): frozen alone is at test Δm +2.1, barely above
  the plain YOLO26-s (+1.4): the ViT-B features are not used well without adapting them. LoRA on qkv and proj
  (0.88 M) brings it to +8.7, level with the fine-tuned ViT-S (+9.4, within seed noise) with 11.6 M instead of
  31.9 M trainable, at 2.0 h vs 0.9 h (`--dino-grad-ckpt`). LoRA on the MLPs too (+1.47 M) changes nothing
  (+8.3; val +8.5 vs +8.3). All three still improved at epoch 24; the 100-epoch runs gained about 12 Δm over 24.
- Weak classes (test AP): bridge 3.6, playground 1.2, wind turbine 15.1. The image-class errors come from these
  classes too: bridge, oil tank and playground images are confused with each other. When the detector finds no
  bridge, the vote falls to another class.

## Next steps

- Classification head (`--cls-head`; 24 epochs: F1 +3.0 / +4.1 on val / test, but mAP −1.9 / −1.4, so val Δm
  +5.6 vs +7.6 without it): LayerNorm + Linear on the
  DINO CLS token and mean patch token, trained with the detector by a multi-label BCE loss (mosaic mixes classes).
  The image class then comes from the head, not from the detections. It targets the bridge / oil tank / playground
  confusions, where the vote from weak detections fails.
- Bigger DINO: ViT-S+/16 (29 M; `--dino vit_small_plus_patch16_dinov3.lvd1689m`) still fits the limits.
  ViT-B/16 (86 M): frozen or with LoRA (Frozen DINO + LoRA).
- 90° rotations and copy-paste of small objects (docs/improvement.md 4); smaller anchor-free strides (a P2 head) for
  objects of 5-10 px.
