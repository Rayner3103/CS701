# USAGE：当前最佳模型（F1_full）

[English](USAGE.md) · **中文**

本文说明 `F1_full_val.zip`（Junkai 最佳的 Phase 1 提交候选）背后的模型是怎么构建的、如何复现，以及如何用它来检验一项改动。环境配置、数据下载和本地划分见 [setup.zh-CN.md](setup.zh-CN.md)（步骤 1-4）；下面所有命令都在仓库文件夹中运行。

## 1. 结果

| 模型 | 训练数据 | Codabench val Δm | macro-F1 | mAP | AP50 |
|---|---|---|---|---|---|
| V2，第 100 轮（`V2_e100_val.zip`） | 本地 train+val（8,452 张图像） | **38.23** | 95.06 | 45.62 | 70.38 |
| F1_full（`F1_full_val.zip`，本文档） | 全部 9,392 张有标签图像 | 尚未提交 | | | |

F1_full 是 V2 配方加两处改动：它在全部有标签图像上训练（包括本地 test 划分，不再有留出集）100 轮，也就是 V2 在其留出集上达到峰值的轮数；并且使用 2.5 节的最终后处理。本地留出集会高估 Codabench 成绩：V2 第 100 轮在本地 test 划分上得分 mAP 48.91 / F1 95.59，在 Codabench val 上却是 45.62 / 95.06。

## 2. 方法

一个模型，每张图像一次前向，同时完成两个任务：

```
gray image, stretched to 512 x 512
  -> DINOv3 ViT-B/16 (frozen) + LoRA          32 x 32 tokens of 768
  -> mean of the tokens -> LayerNorm + Linear  image class (9)
  -> ViTDet simple feature pyramid             strides 4, 8, 16, 32, 64
  -> RPN (NWD matching) -> 3-stage Cascade R-CNN -> boxes, classes, scores
```

### 2.1 骨干
- timm 的 `vit_base_patch16_dinov3.lvd1689m`（Meta DINOv3，在 1.7 B 张通用网络图像上预训练，不含 SAR）。timm 把 RGB patch-embedding 权重相加成一个灰度通道。
- 冻结。训练的部分：全部 12 个 block 的 `attn.qkv`、`attn.proj`、`mlp.fc1`、`mlp.fc2` 上的 LoRA（rank 32，alpha 64），以及所有 LayerNorm（`--train-norms`）。

### 2.2 检测头（`sarbench/tiny.py`）
- **Cascade R-CNN**，3 个阶段（IoU 0.5 / 0.6 / 0.7），轻量的类无关框头，aligned RoIAlign。分数是 3 个阶段类别概率的均值（标准的 Cascade 推理）。
- **RPN 用 NWD 匹配**（normalized Gaussian Wasserstein distance，C = 8），而不是只用 IoU：对偏移几个像素的 8 px 框，IoU 接近零，NWD 则不会。
- **Anchors** 五个层级上为 6, 8, 11 / 16, 19, 23 / 32, 38, 45 / 64, 76, 91 / 128, 152, 181 px，以像素网格为中心（`--anchor-offset 0.5`）。半数的框小于一个 16 px patch。

### 2.3 训练
- Batch 16，AdamW（weight decay 0.05），LoRA lr 3e-4，head lr 1e-4，一轮 warmup，余弦衰减到第 100 轮时为 0，bf16，`torch.compile`，只做翻转（没有其他增强）。
- 2 块 GPU 用 DDP（每块 GPU 每步处理 8 张图像；全局 batch 保持 16）。

### 2.4 规模和计算量（报告必须说明的内容）
- 参数：**总计 109,014,752，可训练 23,805,152**（限制 130 M / 40 M）；`train.py` 会打印这两个数。
- 训练：2 x NVIDIA RTX 6000 Ada（48 GB），F1_full 用时 3.5 h，每块 GPU 峰值 38 GB。
- 推理：一块 GPU 上每张图像约 15 ms（bf16），soft-NMS 在 CPU 上运行。
- 输入：每张图像以 512 x 512 输入一次（非正方形图像被拉伸；不切片，不放大到 512 以上），用 train 集的 mean / std（0.2526 / 0.2133）归一化。

### 2.5 后处理（`pp_eval.FINAL`，`predict.py` 的默认设置）
全部都作用在同一次前向的输出上（没有集成，没有测试时增强）：

| 步骤 | 本地效果 |
|---|---|
| 框分数 = 检测器概率 x RPN objectness^0.25，然后按类做 Gaussian soft-NMS（sigma 0.75） | 约 +1.3 至 +1.8 mAP |
| 框 = 三个 Cascade 阶段的框的均值 | +0.3 至 +0.5 mAP（3 个模型） |
| 框移动 -0.4 px（在本地留出集上测得的系统性标注偏移） | +0.4 至 +0.7 mAP |
| 分数阈值 1e-5，每张图像至少 30 个框，最多 100 个 | +0.06 至 +0.16 mAP |
| 图像类别 = log p_classifier + log（该类别的最高检测器概率）的 argmax | 约 ±0.2 F1 |

### 2.6 各部分的贡献（本地留出集，相对我们运行的参考模型的 Δm，无后处理）

| 运行 | 改动 | 轮数 | 本地 Δm |
|---|---|---|---|
| E00 | 参考：ViT-B/16 IN-21k + LoRA r16 | 24 | 0 |
| E01 | DINOv3 骨干 | 24 | +8.1 |
| E04r | + 所有线性层上的 LoRA r32，训练 norm | 24 | +18.1 |
| D1 / D2 | + Cascade R-CNN / + NWD 匹配和小 anchor | 24 | +21.0 / +19.7 |
| C36 | + 两者 | 36 | +31.3 |
| C48 | 同上，训练更久 | 48 | +38.6 |
| V2 | 同上，在 train+val 上训练 120 轮（只看本地 test） | 100 / 120 | 第 100 轮最佳 |

试过但未采用：重叠的 stride-8 patch（`--token-stride 8`，计算量 2.6 倍，-0.4 mAP：在 bridge / playground / wind turbine 上更好，在 oil tank 上差很多）、LoRA rank 64、D4 旋转 + copy-paste、多层级 ViT neck、ConvNeXt 骨干、Mona adapter。

## 3. 复现 F1_full

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

- 一块 GPU：参数相同，把 `torchrun ...` 换成 `python train.py`（约 7 h，峰值约 43 GB）。
- 保留 `--anchor-sizes` 外面的引号：`;` 会结束 shell 命令。
- 运行结果写入 `runs/F1_full/`：`model.pt`（只含训练的权重，约 100 MB；冻结的 DINOv3 权重从 timm 重新加载）、`metrics.json`（参数、参数量、时间、峰值内存）、`log.txt`，以及 `ckpt_e10.pt` ... `ckpt_e90.pt`。

## 4. 生成并检查提交文件

```bash
# Phase 1 (val); for Phase 2 use ../cs701-sar-course-data/test
python predict.py --run runs/F1_full --images ../cs701-sar-course-data/val --out submission_val.zip \
  --raw-out runs/F1_full/raw_codabench_val.pt
python tools/check_submission.py submission_val.zip ../cs701-sar-course-data/val/images.json
```

- `predict.py` 默认使用最终后处理；`--weights ckpt_e90.pt` 选择一个 checkpoint，`--device cpu` 不用 GPU 运行（每张图像约 0.7 s）。
- `--raw-out` 保留每张图像的原始输出。之后用其他后处理重新打分只需几秒，也不需要模型：在同一命令上加 `--from-raw runs/F1_full/raw_codabench_val.pt`。
- `tools/check_submission.py` 检查 Evaluation 页面的格式规则（文件、表头、每张图像一行、labels 0..8、category_id 1..9、w, h > 0 的有限框），并报告超出图像范围的框，以及框的类别是否与图像类别一致。

## 5. 用当前配方检验一项改动

先在本地划分上筛选（在本地 train 上训练，在本地 val 和 test 上打分），然后与 C36 比较：

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

C36 在最终后处理下的成绩（需要超过的标准）：本地 val mAP 37.22 / macro-F1 95.54，本地 test mAP 39.04 / macro-F1 95.16（一块 GPU 约 2 h）。同一配置单次运行的波动约为 ±1 Δm。

这个分支上新增的常用 `train.py` 选项：`--train-split` / `--eval-splits` / `--eval-every` / `--save-every`（增长曲线和 checkpoint）、`--ema`、`--aug d4`、`--zoom-p`、`--cascade-size-iou`、`--iou-branch`、`--flip-bias`、`--token-stride`、`--unfreeze-last`、`--spatial-prior`；完整列表请运行 `python train.py --help`。
