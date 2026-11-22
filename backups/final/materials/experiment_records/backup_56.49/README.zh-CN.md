# BP7 cos200，第 100–180 轮参数平均，类无关融合：备份与复现

[English](README.md) · **中文**

`submission/BP7_avgE100-180_val_agn.zip` 是队伍 Phase 1 的最终条目：排行榜 val Δm **56.49**（macro-F1 95.27，mAP 55.10），2026-11-01 20:59 SGT 发布，Phase 1 关闭时 8 队中第 2。它是一个模型：一次在全部 9,392 张有标签图上的 200 轮余弦训练，取其 9 个 checkpoint（第 100、110、…、180 轮）的可训练张量平均成一组权重，每张图前向一次，再加上 Phase 1 最后一天调好的后处理链。

本 README 说明文件夹里有什么，以及三种复现方式。数字、解释和报告各项见 [REPORT.zh-CN.md](../../report/REPORT.zh-CN.md)。

## 1. 结果（排行榜，Phase 1 val，1,426 张图）

| 提交 | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|
| F2（10-27） | 44.84 | 95.46 | 48.97 | 71.53 | 95.79 |
| SC2 e130（10-28） | 51.86 | 95.35 | 52.66 | 73.18 | 95.58 |
| BP7 cos160 e130（10-31） | 52.78 | 95.35 | 53.15 | 73.76 | 95.58 |
| BP7 cos200 e180，调优后处理（11-01） | 54.83 | 95.09 | 54.29 | 74.09 | 95.44 |
| 同权重 e100–180 平均，硬过滤 | 56.41 | 95.27 | 55.06 | 74.79 | 95.58 |
| **e100–180 平均，类无关融合（本备份）** | **56.49** | **95.27** | **55.10** | **74.85** | **95.58** |

关闭时的 Phase 1 榜单：Team 9 67.16，**Team 7（我们）56.49**，Team 2 54.81，Team 1 54.24，Team 8 51.03，Team 3 47.75，Team 5 45.89，Team 4 44.50（`run/leaderboard_phase1_public_2026-11-01T2120SGT.json`）。

## 2. 模型

### 2.1 与 SC2（51.86）的区别

| 设置 | SC2 | 本条目 | 对 val Δm 的影响 |
|---|---|---|---|
| `--spatial-prior` | 关 | **64**（作用于原图的 64 通道 CNN 分支，+210,688 参数） | 51.86 → 52.78 的一部分 |
| 余弦日程 | 150 轮，取 e130 | **200 轮**（`--schedule-epochs 200`），每 10 轮存 checkpoint | e160 时 52.78 → 53.41 |
| 使用的权重 | 一个 checkpoint | **e100、110、…、180 的等权平均**（`average_checkpoints.py`） | +0.6 到 +1.0 |
| `--soft-sigma` | 0.75 | **0.5** | +0.29 |
| `--obj-pow`（分数 × RPN objectness^p） | 0.25 | **0** | +1.05 |
| `--rpn-topn`（测试候选数） | 1000 | **2000** | +0.02 |
| `--cls-lambda`（分类与检测头融合） | 1 | **2** | +0.04 |
| `--fusion` | none | **agn**（框分数 = (1 − p_bg) × p_img(类别)，保留全部 9 类） | +0.5 到 +0.7 |
| `--shift` | −0.2 | −0.2 | — |

其余与 SC2 相同：DINOv3 ViT-B/16（冻结）+ `attn.qkv, attn.proj, mlp.fc1, mlp.fc2` 上的 LoRA rank 80 / alpha 160 + 可训练 LayerNorm，特征金字塔 P2–P6，锚框 6–181 px、按 max(IoU, NWD C=8) 匹配的 RPN，宽度 1536、带尺寸自适应 IoU 阈值（A4）的三阶段 Cascade R-CNN，aligned RoIAlign，锚框偏移 0.5，翻转增强配 `--flip-bias -0.3`，AdamW（头 1e-4，骨干 3e-4，weight decay 0.05），1 轮 warm-up，双卡 DDP、总 batch 16，梯度检查点（仅省显存），全部 9,392 张有标签图（本地划分的 `--train-split train+val+test`）。

### 2.2 参数（`run/parameter_breakdown.json`）

| 部分 | 总计 | 可训练 |
|---|---|---|
| DINOv3 ViT-B/16 骨干（冻结） | 85,135,104 | 0 |
| qkv/proj/fc1/fc2 上的 LoRA r80，12 个 block | 11,796,480 | 11,796,480 |
| 骨干 LayerNorm（可训练）及其他一维张量（冻结） | 112,896 | 38,400 |
| 颈部：特征金字塔 5,560,000 + 空间先验 210,688 | 5,770,688 | 5,770,688 |
| RPN | 601,645 | 601,645 |
| Cascade RoI 头，3 阶段，宽度 1536（+ 预测层 64,554） | 21,651,690 | 21,651,690 |
| 图像分类头 | 8,457 | 8,457 |
| **总计** | **125,076,960**（上限 130M） | **39,867,360**（上限 40M，99.7%） |

平均不改变结构和参数量：平均后的文件仍是同样的 234 个可训练张量（39,867,360 个值）。

## 3. 文件夹内容

| 路径 | 内容 | 进 git？ |
|---|---|---|
| `code/` | 代码快照：11-01 21:30 的工作区（git `8445d58` 加未提交的新增）。训练用的是运行目录里的快照；`run/code_diff_training_snapshot_to_backup.diff` 列出全部差异（新的可选参数、一个 LingBot 骨干文件），都不影响本模型。 | 是 |
| `train_BP7_cos200.sh` | 双卡重训这条轨迹（第 180 轮后干净停止，每 10 轮存 checkpoint） | 是 |
| `average_checkpoints.py` | 把 `weights/ckpt_e100..e180.pt` 平均成 `weights/model.pt`（与备份文件逐位相同） | 是 |
| `make_submission.sh` | 用最终后处理生成并检查提交文件；`IMAGES=…/test` 用于 Phase 2；`FROM_RAW=1` 不需要 GPU | 是 |
| `run/args.json`、`run/metrics_e200.json` | train.py 的全部参数；参数量、训练时间、显存峰值 | 是 |
| `run/logs/` | 五段训练日志：第 1 轮（原始启动，第 2 轮显存错误）、2–130、131–147、148–160、161–200 | 是 |
| `run/suite_records/` | 启动清单、checkpoint 与提交校验、最后一天的决策记录 | 是 |
| `run/official_scores_phase1.json` | 项目的每一个官方分数，带日期和设置 | 是 |
| `run/leaderboard_phase1_public_*.json` | 锁定前 40 分钟的 Phase 1 公开榜单 | 是 |
| `run/local_holdout_window_sweep.json`、`run/local_holdout_postprocess_bootstrap_{val,test}.json`、`run/boot_eval.py` | 平均窗口和融合规则的本地证据，以及产生它们的配对 bootstrap 评估器 | 是 |
| `run/averaging_manifest.json`、`run/parameter_breakdown.json`、`run/dataset_stats.json`、`run/gpu_hours_runs_since_2026-10-28.json`、`run/pip_freeze.txt` | 平均的来源（每个源文件的 SHA256）、参数表、数据集统计、GPU 小时估算、包版本 | 是 |
| `submission/` | 56.49 的 zip（类无关融合）和 56.41 的 zip（硬过滤），带 SHA256 | 是 |
| `weights/model.pt` | 平均后的权重（仅可训练张量，160 MB） | 否，仅本机 |
| `weights/ckpt_e100.pt … ckpt_e180.pt` | 9 个源 checkpoint（9 × 160 MB，硬链接到运行目录） | 否 |
| `weights/val_raw.pt` | 1,426 张 val 图上的模型原始输出（675 MB）；`make_submission.sh FROM_RAW=1` 据此重建完全相同的 zip | 否 |
| `weights/metrics.json` | 运行的 metrics.json 副本，让 `predict.py --run weights` 找到模型参数 | 否 |
| `report-page.html` | 项目汇报页面（中英双语、单文件 HTML；也发布为私有的 claude.ai artifact） | 是 |
| `README*.md`、`REPORT*.md` | 本指南和报告素材，中英各一份 | 是 |

在 Junkai 的机器上 `weights/` 位于 `/mnt/hdd/junkaichen/CS701/CS701/backups/56.49-…/weights/`（硬链接指向 `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full9392_cos200_*`）。`weights/SHA256SUMS` 列出了每个文件。

## 4. 复现

### 4.0 三种方式

| 方式 | 需要 | 时间 | 结果 |
|---|---|---|---|
| A. 重算原始输出 | `weights/val_raw.pt`，CPU | 几秒 | 逐字节相同的 zip |
| B. 用平均权重推理 | `weights/model.pt`，1 张 GPU | 约 1 分钟 | 分类相同，框在 GPU 浮点噪声范围内 |
| C. 重训、平均、推理 | 2 张 GPU，数据 | 约 13 小时 + 1 分钟 | 一条新轨迹：分数波动约 ±0.5 Δm |

### 4.1 环境与数据

与 F2 相同：`../44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/README.zh-CN.md` 的 4.1–4.2 节（conda 环境 `cs701bench`，torch 2.9.0+cu128，timm 1.0.24；`prepare_course_data.py` 把 Hugging Face 下载转换成 `dataset/SARFact-Course-20K`）。`run/pip_freeze.txt` 有精确的包版本。以下命令都在本文件夹里运行。

### 4.2 生成提交文件（方式 A 和 B）

```bash
# A：无需 GPU，重算保存的原始输出（zip 内容完全相同）
FROM_RAW=1 IMAGES=/path/to/cs701-sar-course-data/val ./make_submission.sh

# B：用平均权重推理（1 张 GPU，1,426 张约 1 分钟）
IMAGES=/path/to/cs701-sar-course-data/val ./make_submission.sh

# Phase 2：test 图（方式 B；test 没跑过一次之前没有原始输出文件）
IMAGES=/path/to/cs701-sar-course-data/test ./make_submission.sh
```

脚本运行 `predict.py --run weights --weights model.pt --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --stage-box mean --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1 --topk 9`，然后运行 `tools/check_submission.py`。在与其他任务共用的 GPU 上，可加前缀 `python tools/train_gpu_limited.py --memory-fraction 0.15 predict.py …`（历史上就是这样跑的）限制显存。

### 4.3 平均 checkpoint

```bash
python average_checkpoints.py                      # weights/ckpt_e100..e180.pt -> weights/model.pt（拒绝覆盖）
python average_checkpoints.py --epochs 100 120 140 160 --out other_window.pt
```

等权 1/9，float64 累加，float32 输出；`<out>.json` 记录每个源文件的 SHA256。

### 4.4 从头训练（方式 C）

```bash
DATA=/path/to/dataset/SARFact-Course-20K ./train_BP7_cos200.sh      # 2 张 GPU；第 180 轮后停止
python average_checkpoints.py --dir runs/BP7_cos200 --out runs/BP7_cos200/avg_e100-180.pt
cp runs/BP7_cos200/metrics.json weights/ 2>/dev/null; # 或把 make_submission.sh 的 RUN 指向 runs/BP7_cos200
```

- 原始运行：2 × RTX 6000 Ada，与其他任务共用，13.0 小时到第 180 轮（14.14 小时到 200），开 `--grad-ckpt` 时每卡峰值 12.1 GB，独占 GPU 后每卡约 25 张/秒（每轮 3 分 40 秒）。在空闲的 48 GB GPU 上可去掉 `--grad-ckpt` 提速；数学结果相同。
- 日志开头应为 `parameters: 125,076,960 total, 39,867,360 trainable`。
- 运行在第 180 轮后干净停止，学习率日程仍是 200 轮的（`--stop-after-epoch` 不缩短日程）。原始运行继续到了第 200 轮；e190/e200 未被使用。
- 原始运行中断过三次（第 131 轮续训时 RPN 的一次分配超过了每进程显存上限；第 147 轮 `/tmp` 写满）。每次都恢复了完整的 `training_state.pt`（模型、AdamW 动量、日程步数、两个 rank 的随机状态）并验证了下一轮，所以轨迹与一次不中断的运行相同。续训：`./train_BP7_cos200.sh --resume runs/BP7_cos200/training_state.pt`。
- DDP 和 cuDNN 在 GPU 上不是逐位确定的：重训得到的权重和分数略有不同（单个 checkpoint 在榜上约 ±0.5 Δm；平均会减小这个波动）。

### 4.5 2026-11-01 做过的核对

- 方式 A 用 `weights/val_raw.pt`：zip 里的 `classification.csv` 和 `detection.json` 与 `submission/BP7_avgE100-180_val_agn.zip` 逐字节相同。
- 对备份的 9 个 checkpoint 运行 `average_checkpoints.py`：全部 234 个张量与 `weights/model.pt` 逐位相同（文件哈希仅因序列化不同）。
- 方式 B，batch 16，在一张同时跑训练任务的 GPU 上：分类相同；59,668 个框而非 59,711（bf16 的 batch 大小效应让少数框跨过分数阈值）；每张 23.2 ms。
- 比较 zip 里的文件，不要比较 zip 的 SHA256（zip 记录了文件时间）。

### 4.6 复现本地证据

平均窗口和融合规则在一条只用 7,512 张本地划分训练的独立轨迹（`cs701_runs/BP7_LR130_local_s0_20261029_v2/lr1x`，每 10 轮存 checkpoint）上，用两个有标签留出集（各 940 张）核对过：

```bash
# 用最终后处理（不融合 / hard / agn）在有标签的本地划分上给一个权重文件评分
python code/predict.py --run <local run> --weights <ckpt 或平均后的 .pt> --data /path/to/dataset/SARFact-Course-20K \
    --split val --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1 --topk 9
# 用一份原始输出（上面加 --dump-raw 会写出 raw_<split>.pt）对 13 种后处理变体做配对 bootstrap
python run/boot_eval.py val <raw_val.pt> 500        # 换机器时修改文件开头的 ROOT
```

结果：`run/local_holdout_window_sweep.json` 和 `run/local_holdout_postprocess_bootstrap_{val,test}.json`；REPORT.zh-CN.md 的 5.2–5.3 节有总结。

### 4.7 常见问题

- `predict.py --run weights` 需要 `weights/metrics.json`（模型参数）；文件夹里有。
- `--from-raw` 忽略 `--rpn-topn`（原始文件里每张图已有 2,000 个候选）。
- 共用 GPU 上续训开头的显存错误：提高 `tools/train_gpu_limited.py` 的显存比例（GPU 归我们独占后 0.2 太低；0.35 可用）。
- 数据加载进程因 "No space left on device" 卡住：启动前把 `TMPDIR` 设到有空间的磁盘（11-01 的运行在第 147 轮遇到过）。
