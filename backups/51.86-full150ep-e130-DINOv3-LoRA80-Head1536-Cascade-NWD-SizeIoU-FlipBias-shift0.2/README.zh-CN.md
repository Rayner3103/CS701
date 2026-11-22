# SC2 第 130 轮（全量，参数放大）备份与复现

[English](README.md) · **中文**

`submission/SC2_e130_val_shift-0.2.zip` 是目前最好的 Phase 1 提交：新排行榜 val Δm **51.86**，比 F2（44.84）高 7.0。模型是 F2 的配方放大到参数预算上限（可训练 39.7M / 40M），用全部 9,392 张有标注图训练 150 轮，取第 130 轮的检查点。

这份 README 说明文件夹里有什么、怎样复现。写报告用的数字和解读见 [REPORT.zh-CN.md](REPORT.zh-CN.md)。

## 1. 结果（新排行榜 Phase 1 val，2026-10-28）

| 提交 | 训练 | Δm | macro-F1 | mAP | AP50 | Acc |
|---|---|---|---|---|---|---|
| F2（shift −0.2） | 23.8M 可训练，100 轮 | 44.84 | 95.46 | 48.97 | 71.53 | 95.79 |
| **SC2 第 130 轮（本备份）** | **39.7M 可训练，150 轮中的第 130 轮** | **51.86** | 95.35 | **52.66** | 73.18 | 95.58 |
| SC2 第 150 轮 | 同上，最后一轮 | 51.81 | 95.44 | 52.61 | 73.24 | 95.72 |

- 比 F2 高 7.0 Δm，全部来自检测：mAP +3.7，AP50 +1.7，F1 基本不变。
- 第 130 轮和第 150 轮只差 0.05 Δm，远小于随机波动（约 ±0.5），说明 130 轮左右已经进入平台期。

## 2. 和 F2 的区别

| 参数 | F2 | SC2 | 作用 |
|---|---|---|---|
| `--lora-rank` / `--lora-alpha` | 32 / 64 | **80 / 160** | LoRA 容量 2.5 倍，α/r 保持 2 |
| `--cascade-rep` | 1024 | **1536** | 三个 Cascade 阶段的全连接层宽 1.5 倍 |
| `--epochs` | 100 | **150**（取第 130 轮） | 余弦学习率按 150 轮衰减 |
| `--grad-ckpt` | 无 | **有** | 只省显存（两张卡上各有别人约 25 GB 的进程），计算结果不变 |

其他完全相同：全部 9,392 张、两卡 DDP、全局 batch 16、A4（`--cascade-size-iou`）、A2（`--flip-bias -0.3`）、最终后处理加 −0.2 像素平移。

### 参数量

| 组成 | F2 可训练 | SC2 可训练 |
|---|---|---|
| DINOv3 ViT-B/16 骨干（冻结） | 0（总 85,209,600） | 0（总 85,209,600） |
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
| `code/` | 训练时的代码快照，和 F2 用的代码（commit `51257a2`）逐文件相同 | 是 |
| `train_SC2_full150.sh` | 重新训练（两卡，150 轮，每 10 轮存检查点） | 是 |
| `make_submission_SC2_e130.sh` | 用第 130 轮的权重生成并检查提交文件 | 是 |
| `run/args.json`，`run/metrics.json` | train.py 的全部参数、参数量、训练时长、显存峰值 | 是 |
| `run/log.txt`，`run/SC2_full150_run.sh` | 完整训练日志，和当时实际执行的脚本 | 是 |
| `run/pip_freeze.txt` | conda 环境 `cs701bench` 的包版本 | 是 |
| `submission/` | 提交文件（51.86）和 SHA256 | 是 |
| `weights/model.pt` | 第 130 轮的权重（就是 `ckpt_e130.pt`，只存可训练部分，159 MB） | 否，只在本机 |
| `weights/SC2_e130_val_raw.pt` | Phase 1 val 图片的原始模型输出 | 否，只在本机 |
| `README*.md`，`REPORT*.md` | 本说明和报告素材，中英各一份 | 是 |

`weights/` 在本机的位置是 `/mnt/hdd/junkaichen/CS701/CS701/backups/51.86-full150ep-e130-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2/weights/`。完整训练目录 `/mnt/hdd/junkaichen/CS701/cs701_runs/SC2_full150/` 里有第 10 到 140 轮的全部检查点，以及第 150 轮的 `model.pt`。

## 4. 复现

环境和数据的准备和 F2 一样，见 `../44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/README.zh-CN.md` 第 4.1–4.2 节。下面的命令在本文件夹里运行。

### 4.1 生成提交文件

```bash
# 用第 130 轮的权重推理（1 张 GPU，约 1 分钟）
IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_SC2_e130.sh

# 不用 GPU：从保存的原始输出重算后处理（几秒）
FROM_RAW=1 IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_SC2_e130.sh

# Phase 2：换成 test 图片
IMAGES=/path/to/cs701-sar-course-data/test ./make_submission_SC2_e130.sh
```

2026-10-28 核对：
- 用 `weights/model.pt` 重新推理：zip 里的 `classification.csv` 和 `detection.json` 都和 `submission/` 里的逐字节相同。
- `FROM_RAW=1`：`classification.csv` 相同；`detection.json` 的框最多差 0.1 像素、分数最多差 0.003。原始输出是以较低精度保存的，这个差别对 mAP 可以忽略。
- 比较时比 zip 里的两个文件，不要比 zip 本身的 SHA256（zip 记录了文件时间）。

### 4.2 从头训练

```bash
DATA=/path/to/dataset/SARFact-Course-20K ./train_SC2_full150.sh
```

- 原始运行：2 张 RTX 6000 Ada，6.38 小时，每卡显存峰值 10.6 GB（开了梯度检查点），每卡约 31 张/秒。
- 日志开头应是 `parameters: 124,866,272 total, 39,656,672 trainable`。
- 第 130 轮的权重是 `runs/SC2_full150/ckpt_e130.pt`。注意它是 150 轮余弦学习率中途的检查点，不等于“训 130 轮”的结果；要复现必须按 150 轮的设置训练。
- DDP、`torch.compile` 和 cuDNN 在 GPU 上不是逐位确定的，重新训练的分数会有波动（排行榜上约 ±0.5 Δm）。
- 显存够（每卡约 40 GB 以上）的话，可以去掉 `--grad-ckpt`，会快不少，结果不变。
