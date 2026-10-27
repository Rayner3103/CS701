# F2_full（shift −0.2）备份

`submission/F2_full_val_shift-0.2.zip`（12,363,762 字节）是 Junkai 分支目前推荐的 Phase 1 提交文件。本文件夹收齐了复现它需要的所有东西：训练和推理的代码、命令、参数、日志、环境版本和这份提交文件。

## 1. 结果

| 文件 | 内容 | Codabench val Δm |
|---|---|---|
| `F1_full_val.zip` | v2 配方，在全部 9,392 张有标注图上训练 100 轮 | 41.26（F1 95.90，mAP 46.97） |
| **`F2_full_val_shift-0.2.zip`**（本备份） | F1 + A4 + A2，推理时把所有框平移 −0.2 像素 | **44.84**（F1 95.46，mAP 48.97，AP50 71.53，Acc 95.79） |

F2 比 F1 高 3.58 Δm：mAP +2.00，macro-F1 −0.44。

本地消融用 36 轮的 C36 配方做，基准是 36 个随机种子的平均（val / test mAP 37.46±0.35 / 38.90±0.37）。在这个基准上，A4 + A2 一起用得到 val / test mAP 38.43 / 40.01。−0.2 平移在两个本地留出集上各多 0.2 mAP。

## 2. 方法

- **骨干**：DINOv3 ViT-B/16（timm `vit_base_patch16_dinov3.lvd1689m`，通用图片预训练，没用 SAR 数据）。骨干权重冻结，只训练 LoRA（rank 32，alpha 64，加在所有 block 的 qkv / proj / fc1 / fc2 上）和 LayerNorm。
- **检测**：3 阶段 Cascade R-CNN。RPN 用 NWD 匹配（C = 8），最小锚框 6 像素，RoIAlign 对齐到像素中心。
- **A4**（`--cascade-size-iou`，表 `a4`）：三个 cascade 阶段的 IoU 阈值随目标大小调整，小目标的阈值放宽。
- **A2**（`--flip-bias -0.3`）：数据标注的框整体有一点偏移，翻转增强会把这个偏移翻到反方向。A2 在翻转时把框按测得的偏差挪回来，让翻转后的框和原图的标注习惯一致。
- **训练**：全部 9,392 张有标注图（`--train-split train+val+test`），100 轮，余弦学习率，bf16，`torch.compile`，2 卡 DDP，全局 batch 16。
- **后处理**（`predict.py` 默认值）：框分数乘以 RPN objectness^0.25，按类 soft-NMS；框取三个阶段的平均；分数阈值 1e-5，每张图至少保留 30 个框；图像分类结合检测结果。最后加 `--shift -0.2`，所有框平移 −0.2 像素。
- **规模和规则**：总参数 109,014,752，可训练 23,805,152（上限 130M / 40M）。单模型，不用 TTA，每张图只以 512×512 输入一次。

## 3. 文件夹内容

| 路径 | 内容 | 在 git 里吗 |
|---|---|---|
| `code/` | 训练时的代码快照（`cs701_runs/F2_full/code`），和 commit `51257a2` 完全一致 | 是 |
| `train_F2_full.sh` | 重新训练 F2 的命令 | 是 |
| `make_submission_F2.sh` | 生成并检查 shift −0.2 的提交文件 | 是 |
| `run/args.json`，`run/metrics.json` | train.py 的全部参数、参数量、训练时长、显存峰值 | 是 |
| `run/log.txt` | 完整训练日志 | 是 |
| `run/a3a1_F2_full_ddp.sh`，`run/F2_extra_args.txt` | 当时队列里实际执行的原始脚本（路径是本机的） | 是 |
| `run/pip_freeze.txt` | conda 环境 `cs701bench` 的包版本（torch 2.9.0+cu128，torchvision 0.24.0，timm 1.0.24） | 是 |
| `submission/` | 提交文件和 SHA256 | 是 |
| `weights/model.pt` | 训练好的权重（只存可训练部分，95 MB；冻结的 DINOv3 权重运行时从 timm 下载） | 否，只在本机 |
| `weights/F2_full_val_raw.pt` | Codabench val 图片的原始模型输出（338 MB） | 否，只在本机 |

权重没有放进 git：GitHub 单个文件限 100 MB，而且一旦提交，所有人 clone 都要下载它。本机的位置是 `/mnt/hdd/junkaichen/CS701/CS701/backups/44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/weights/`，原始训练目录 `/mnt/hdd/junkaichen/CS701/cs701_runs/F2_full/` 里还有一份，外加 `ckpt_e10` 到 `ckpt_e90`。

## 4. 复现

详细步骤（环境、数据、训练参数逐项说明、核对方法、本地消融）见 [REPRODUCE.md](REPRODUCE.md)。方法是怎样一步步优化到 F2 的、每一步的取舍，见 [METHOD_EVOLUTION.md](METHOD_EVOLUTION.md)。写报告用的数字、表格和注意事项，见 [REPORT.md](REPORT.md)。下面是最简流程。

环境和数据准备见仓库根目录的 `setup.md`（Step 1–4）。两个脚本默认按 `setup.md` 的目录结构找数据，数据放在别处时用 `DATA`（转换后的数据）和 `IMAGES`（Codabench 图片目录）指定。

```bash
conda activate cs701bench

# 1) 训练（2 卡，约 3.6 小时，每卡峰值 38 GB），输出在 runs/F2_full/
DATA=/path/to/dataset/SARFact-Course-20K ./train_F2_full.sh

# 2) 生成提交文件并检查格式（默认用 weights/，没有就用 runs/F2_full）
IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_F2.sh
#    Phase 2：IMAGES 改成 .../cs701-sar-course-data/test
#    不用 GPU：FROM_RAW=1 ./make_submission_F2.sh，从 weights/F2_full_val_raw.pt 重算后处理，几秒钟
```

在本机上 `DATA=/mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K`，`IMAGES=/mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/val`。

## 5. 已验证（2026-10-27）

- `code/` 和仓库 commit `51257a2` 的代码逐文件相同。
- `FROM_RAW=1` 重新生成的 zip，里面 `classification.csv` 和 `detection.json` 与 `submission/` 里的文件逐字节相同。zip 文件本身的 SHA256 不同，因为 zip 里记录了文件时间。
- 用 `weights/model.pt` 在 GPU 上重新推理：分类结果完全相同，框数相同（79,969），框分数最大差 0.0024，是 GPU 浮点运算的正常误差。
- 重新训练没有做。随机种子固定为 0，但 DDP 和 `torch.compile` 在 GPU 上不是逐位确定的，重新训练的结果会在种子噪声范围内波动（C36 上约 ±0.35 mAP）。
