# SC1（80:20，参数放大）备份与复现

[English](README.md) · **中文**

SC1 是把 F2 的配方在参数预算内放大后，在本地 80:20 划分上训练 36 轮的版本：本地 train 7,512 张，在本地 val 和 test 各 940 张上评估。文件夹名里的 28.90 是它在新排行榜 Phase 1 val 上的 Δm（和 F2 文件夹一样用排行榜分数命名）；它的本地 Δm 是 43.75。

**28.90 比 F2 的 44.84 低很多，但这是预料中的，不能说明放大参数没用。** 这份和 F2 的差别主要在训练数据和轮数（80% 数据、36 轮 vs 全部数据、100 轮），不在参数量。见第 1.2 节。

这份 README 说明文件夹里有什么、怎样复现。为什么这样放大、结果怎么解读，见 [REPORT.zh-CN.md](REPORT.zh-CN.md)。同样配置的全量训练（SC2，150 轮）已经训完，见 REPORT 第 5 节。

## 1. 结果

### 1.1 本地留出集

本地划分和 F2 的本地消融相同（seed 0），36 轮，最终后处理（不平移）。本地 Δm 相对本地参考 E00，取 val 和 test 的平均。

| 模型 | 可训练 / 总参数 | val F1 / mAP / AP50 | test F1 / mAP / AP50 | 本地 Δm |
|---|---|---|---|---|
| C36（37 次运行） | 23.8M / 109.0M | 95.39 / 37.46 / — | 95.21 / 38.91 / — | 35.56 ± 0.70 |
| AA（F2 的配方） | 23.8M / 109.0M | 95.05 / 38.43 / 65.85 | 95.55 / 40.01 / 68.27 | 37.83 |
| **SC1（放大）** | **39.7M / 124.9M** | **95.65 / 40.96 / 67.96** | 95.05 / **42.87** / **70.25** | **43.75** |
| SC1 − AA | +15.9M | +0.60 / +2.53 / +2.11 | −0.50 / +2.86 / +1.98 | **+5.92** |

- val 和 test 的 Δm 分别 +5.97 和 +5.87，两边一致。
- +5.92 约是 C36 种子标准差（0.70）的 8.5 倍。SC1 只有 1 个种子，但这个幅度不太可能是运气。
- 再加 −0.2 像素平移：mAP 变成 41.21 / 43.17（+0.25 / +0.30），和 F2 的结论一致。

### 1.2 排行榜（Phase 1 val，2026-10-27 20:52 SGT，未 post）

| 提交 | 训练数据 | 轮数 | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|---|---|
| **SC1（本备份，shift −0.2）** | 80%（7,512 张） | 36 | **28.90** | 94.69 | 40.85 | 67.64 | 94.95 |
| V2 第 100 轮（Codabench） | 90%（8,452 张） | 100 | 38.23 | 95.06 | 45.62 | 70.38 | — |
| F2（shift −0.2） | 100%（9,392 张） | 100 | 44.84 | 95.46 | 48.97 | 71.53 | 95.79 |

为什么比 F2 低这么多：

- **mAP 决定了 Δm。** 参考模型 mAP 只有 26.13，mAP 差 1 个点 Δm 就差约 1.9。SC1 比 F2 低 8 个 mAP，Δm 就差了约 16。
- **轮数影响最大。** 以前本地实验里，同一配方从 48 轮训到 100 轮，test mAP 从 41.5 涨到 48.9。SC1 只训了 36 轮，离平台期还很远。
- **本地分数会高估排行榜。** SC1 本地 test mAP 42.87，排行榜 40.85，低约 2 个点；V2 当时低约 3 个点，所以这次的落差是正常的。

放大到底有没有用，最公平的对照是 AA（F2 原配方，同样 36 轮、80% 数据）。AA 没有交过排行榜，按同样约 2 个点的落差估算：本地 test mAP 40.01 → 排行榜约 38 mAP，Δm 约 24。SC1 是 28.90，约高 5，和本地的 +5.92 基本一致。**这是估算，不是实测。**

这份提交不应该 post：榜上应该挂分数最高的那份（目前是 F2 的 44.84）。

## 2. 和 F2 配方的区别

只改了三个参数，其他和 AA（F2 配方的 36 轮本地版本）完全相同：

| 参数 | AA / F2 | SC1 | 作用 |
|---|---|---|---|
| `--lora-rank` / `--lora-alpha` | 32 / 64 | **80 / 160** | LoRA 容量 2.5 倍，缩放系数 α/r 保持 2 |
| `--cascade-rep` | 1024 | **1536** | 三个 Cascade 阶段的全连接层宽 1.5 倍 |
| `--grad-ckpt` | 无 | **有** | 只省显存（GPU0 和别人共用），计算结果不变，batch 仍是 16 |

### 参数量

| 组成 | F2 可训练 | SC1 可训练 |
|---|---|---|
| DINOv3 ViT-B/16 骨干（冻结，不计入可训练） | 0（总 85,209,600） | 0（总 85,209,600） |
| LoRA | 4,718,592 | **11,796,480** |
| 骨干中的 LayerNorm | 38,400 | 38,400 |
| 特征金字塔 | 5,560,000 | 5,560,000 |
| RPN | 601,645 | 601,645 |
| Cascade RoI 头（3 阶段） | 12,878,058 | **21,651,690** |
| 分类头 | 8,457 | 8,457 |
| **可训练合计**（上限 40M） | 23,805,152 | **39,656,672**（99.1%） |
| **总参数**（上限 130M） | 109,014,752 | **124,866,272** |

## 3. 文件夹内容

| 路径 | 内容 | 在 git 里吗 |
|---|---|---|
| `code/` | 训练时的代码快照（`cs701_runs/SC1_r80h1536/code`），和 F2 用的代码（commit `51257a2`）逐文件相同 | 是 |
| `train_SC1_8020.sh` | 重新训练 SC1 并评估 | 是 |
| `eval_SC1.sh` | 用保存的原始输出重算本地 val / test 分数（CPU，几分钟） | 是 |
| `run/args.json`，`run/metrics.json` | train.py 的全部参数、参数量、训练时长、显存峰值、训练中每 4 轮的评估 | 是 |
| `run/metrics_pp.json` | 最终后处理下的本地 val / test 结果（含每类 AP） | 是 |
| `run/size_ap_vs_AA.json` | 按目标尺寸的 AP，SC1 和 AA 对比 | 是 |
| `run/log.txt`，`run/SC1_run.sh` | 完整训练日志，和当时实际执行的脚本 | 是 |
| `run/pip_freeze.txt` | conda 环境 `cs701bench` 的包版本 | 是 |
| `weights/model.pt` | 训练好的权重（只存可训练部分，159 MB） | 否，只在本机 |
| `weights/raw_val.pt`，`weights/raw_test.pt` | 本地 val / test 的原始模型输出（各 204 MB） | 否，只在本机 |
| `submission/` | 排行榜 val 提交文件（`SC1_8020_val_shift-0.2.zip`，18 MB，28.90）和 SHA256 | 是 |
| `README*.md`，`REPORT*.md` | 本说明和报告素材，中英各一份 | 是 |

`weights/` 在本机的位置是 `/mnt/hdd/junkaichen/CS701/CS701/backups/28.90-8020-36ep-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2/weights/`，原始训练目录 `/mnt/hdd/junkaichen/CS701/cs701_runs/SC1_r80h1536/` 里还有一份。

## 4. 复现

环境和数据的准备和 F2 一样，见 `../44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/README.zh-CN.md` 第 4.1–4.2 节。下面的命令在本文件夹里运行。

### 4.1 只重算分数（不用 GPU）

```bash
./eval_SC1.sh     # 读 weights/raw_{val,test}.pt，写 weights/metrics_pp.json
```

2026-10-27 核对：重算出的 `metrics_pp.json` 和 `run/metrics_pp.json` 完全相同。

### 4.2 从头训练

```bash
DATA=/path/to/dataset/SARFact-Course-20K ./train_SC1_8020.sh
```

- 原始运行：1 张 RTX 6000 Ada，2.64 小时，显存峰值 12.3 GB（开了梯度检查点）。速度在每秒 13–31 张之间，因为 GPU0 同时有别人的进程。
- 显存够（约 45 GB 以上）的话，可以去掉 `--grad-ckpt`，会快不少，结果不变。
- 日志开头应是 `parameters: 124,866,272 total, 39,656,672 trainable`。
- 结束时自动运行 `pp_eval.py`，把 `runs/SC1_r80h1536/metrics_pp.json` 和 `run/metrics_pp.json` 对比。DDP、`torch.compile` 和 cuDNN 在 GPU 上不是逐位确定的，重新训练的分数会有波动（C36 上约 ±0.35 mAP）。

### 4.3 生成排行榜提交文件

```bash
cd code
python predict.py --run ../weights --images /path/to/cs701-sar-course-data/val --shift -0.2 --out SC1_8020_val_shift-0.2.zip
python tools/check_submission.py SC1_8020_val_shift-0.2.zip /path/to/cs701-sar-course-data/val/images.json
```

原文件在 `submission/`；新旧 zip 比较时要比里面的 `classification.csv` 和 `detection.json`，不要比 zip 本身的 SHA256。

### 4.4 复现对照组 AA

去掉 `train_SC1_8020.sh` 里的 `--lora-rank 80 --lora-alpha 160 --cascade-rep 1536 --grad-ckpt`，改回 `--lora-rank 32 --lora-alpha 64`，就是 AA。
