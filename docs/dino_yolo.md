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

- `--yolo yolo26s` swaps YOLO12-s for YOLO26-s (9.95 M: NMS-free end-to-end head, no DFL). DINO + YOLO26-s is 31.7 M
  in total, all trainable. The DINO features go to layers 4, 6 and 10, the backbone outputs the YOLO26 head reads
  (`neck_inputs`; YOLO12: 4, 6, 8).
- `--yolo yolo26s-p2` adds a P2/4 output for the 5-10 px objects (DINO also goes to layer 2). It starts from
  `yolo26s.pt` (`--pretrained`), but only 6.1 of its 9.7 M YOLO parameters load: the backbone and the top-down neck.
  The P2 path and the Detect head start random.
- `--dino none` trains the plain YOLO with the same protocol: the baseline for the DINO fusion.
- The protocol is the YOLO12 one (AdamW, not YOLO26's default MuSGD), so the rows compare. The end-to-end head gives
  at most 100 boxes per image without NMS (`iou` has no effect).
- Not done: a frozen DINO, which would let DINO + YOLO26-m (43.5 M) under the 40 M trainable limit. The ultralytics
  trainer turns gradients back on for every parameter outside `model.N.`, so it needs a trainer change.

```bash
python train_yolo.py --yolo yolo26s --dino none --epochs 24 --name yolo26s_ep24      # plain YOLO26-s
python train_yolo.py --yolo yolo26s --epochs 24 --name dino_yolo26s_ep24             # DINO + YOLO26-s
python train_yolo.py --yolo yolo26s-p2 --epochs 24 --name dino_yolo26s-p2_ep24       # DINO + YOLO26-s-p2
```

## Results

Local split, seed 0, %. Δm as in train.py and compare_runs.py (reference 93.31 / 26.13), so the rows are comparable
with `runs/`. With the test reference of the leaderboard (93.99 / 25.41), dino_yolo12s_ep24 is at about +8.8.

| run | epochs | params (total / trainable) | val F1 | val mAP | test F1 | test mAP | test Δm |
|---|---:|---:|---:|---:|---:|---:|---:|
| dino_yolo12s_ep24 (best.pt = epoch 24) | 24 | 31.2 M / 31.2 M | 91.1 | 30.7 | 90.4 | 30.8 | +7.4 |
| dino_yolo12s_cls_ep24 (`--cls-head`, best.pt = epoch 24) | 24 | 31.2 M / 31.2 M | 94.1 | 28.9 | 94.5 | 29.4 | +6.9 |
| dino_yolo12s_ep50 (best.pt = epoch 49) | 50 | 31.2 M / 31.2 M | 92.2 | 33.9 | 92.2 | 34.3 | +15.0 |
| dino_yolo12s_ep100 (best.pt = epoch 100) | 100 | 31.2 M / 31.2 M | 92.9 | 37.2 | 92.4 | 36.3 | +19.0 |
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
  ViT-B/16 (86 M) needs a frozen or LoRA DINO.
- 90° rotations and copy-paste of small objects (docs/improvement.md 4); smaller anchor-free strides (a P2 head) for
  objects of 5-10 px.
