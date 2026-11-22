# F2_full（shift −0.2）备份与复现

[English](README.md) · **中文**

`submission/F2_full_val_shift-0.2.zip`（12,363,762 字节）是 Junkai 分支目前最好的 Phase 1 提交，新排行榜 val Δm **44.84**。本文件夹收齐了复现它需要的所有东西：代码、命令、参数、日志、环境版本和这份提交文件。

这份 README 说明文件夹里有什么、怎样复现。写报告用的数字、方法演进和取舍见 [REPORT.zh-CN.md](REPORT.zh-CN.md)。

## 1. 结果

| 提交 | 内容 | 平台 | Δm | macro-F1 | mAP | AP50 |
|---|---|---|---|---|---|---|
| `F1_full_val.zip` | v2 配方，在全部 9,392 张有标注图上训练 100 轮 | Codabench val（已停用） | 41.26 | 95.90 | 46.97 | 71.57 |
| **`F2_full_val_shift-0.2.zip`**（本备份） | F1 + A4 + A2，推理时把所有框平移 −0.2 像素 | 新排行榜 Phase 1 val | **44.84** | 95.46 | 48.97 | 71.53 |

- F2 比 F1 高 3.58 Δm：mAP +2.00，macro-F1 −0.44。
- 本地消融：C36 配方（36 轮）的 37 次运行，val / test mAP 是 37.46±0.35 / 38.91±0.36。在同样的设置下，A4 + A2 一起开得到 38.43 / 40.01。−0.2 平移在两个本地留出集上各多 0.2 mAP。

## 2. 方法

- **骨干**：DINOv3 ViT-B/16（timm `vit_base_patch16_dinov3.lvd1689m`，通用图片预训练，没用 SAR 数据）。骨干冻结，只训练 LoRA（rank 32，alpha 64，加在所有 block 的 qkv / proj / fc1 / fc2 上）和 LayerNorm。
- **检测**：3 阶段 Cascade R-CNN。RPN 用 NWD 匹配（C = 8），最小锚框 6 像素，RoIAlign 对齐到像素中心。
- **A4**（`--cascade-size-iou`，表 `a4`）：三个 cascade 阶段的 IoU 阈值随目标大小调整，小目标的阈值放宽。
- **A2**（`--flip-bias -0.3`）：标注的框整体有一点偏移，翻转增强会把它翻到反方向。A2 在翻转时把框按测得的偏差挪回来，让所有训练目标保持同一种坐标约定。
- **训练**：全部 9,392 张有标注图，100 轮，余弦学习率，bf16，`torch.compile`，2 卡 DDP，全局 batch 16。
- **后处理**（`predict.py` 默认值）：框分数乘以 RPN objectness^0.25，按类 soft-NMS；框取三个阶段的平均；分数阈值 1e-5，每张图至少 30 个框；图像分类结合检测结果。另外加 `--shift -0.2`，所有框平移 −0.2 像素。
- **规模和规则**：总参数 109,014,752，可训练 23,805,152（上限 130M / 40M）。单模型，不用 TTA，每张图只以 512×512 输入一次。

## 3. 文件夹内容

| 路径 | 内容 | 在 git 里吗 |
|---|---|---|
| `code/` | 训练时的代码快照（`cs701_runs/F2_full/code`），和 commit `51257a2` 逐文件相同 | 是 |
| `train_F2_full.sh` | 重新训练 F2 | 是 |
| `make_submission_F2.sh` | 生成并检查 shift −0.2 的提交文件 | 是 |
| `run/args.json`，`run/metrics.json` | train.py 的全部参数、参数量、训练时长、显存峰值 | 是 |
| `run/log.txt` | 完整训练日志 | 是 |
| `run/a3a1_F2_full_ddp.sh`，`run/F2_extra_args.txt` | 当时队列里实际执行的原始脚本（路径是本机的） | 是 |
| `run/pip_freeze.txt` | conda 环境 `cs701bench` 的包版本 | 是 |
| `submission/` | 提交文件和 SHA256 | 是 |
| `weights/model.pt` | 训练好的权重（只存可训练部分，95 MB） | 否，只在本机 |
| `weights/F2_full_val_raw.pt` | Phase 1 val 图片的原始模型输出（338 MB） | 否，只在本机 |
| `README*.md`，`REPORT*.md` | 本说明和报告素材，中英各一份 | 是 |

权重没有放进 git：GitHub 单个文件限 100 MB，而且一旦提交，所有人 clone 都要下载它。本机位置是 `/mnt/hdd/junkaichen/CS701/CS701/backups/44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/weights/`；原始训练目录 `/mnt/hdd/junkaichen/CS701/cs701_runs/F2_full/` 里还有一份，外加 `ckpt_e10` 到 `ckpt_e90`。

## 4. 复现

下面的命令都在本文件夹里运行。

### 4.0 三种复现方式

| 方式 | 需要什么 | 耗时 | 结果 |
|---|---|---|---|
| A. 只重算后处理 | 本机 `weights/F2_full_val_raw.pt`，不用 GPU | 几秒 | zip 里的两个文件和原提交逐字节相同 |
| B. 用训练好的权重推理 | 本机 `weights/model.pt`，1 张 GPU | 约 1 分钟 | 分类相同，框数相同，框分数差 ≤0.003 |
| C. 从头训练 | 2 张约 48 GB 的 GPU | 约 3.6 小时 + 推理 | 在随机波动范围内一致（见 4.6） |

`weights/` 只在 Junkai 的机器上，别的机器上只能用方式 C。

### 4.1 环境

```bash
conda create -n cs701bench python=3.12 -y
conda activate cs701bench
pip install -r code/requirements.txt   # torch 2.9.0+cu128, torchvision 0.24.0, timm 1.0.24, pycocotools 2.0.11
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.9.0+cu128 True
```

- 当时环境的完整包列表在 `run/pip_freeze.txt`。
- 原训练硬件：2 × NVIDIA RTX 6000 Ada（48 GB），CUDA 12.8。
- 第一次运行会从 Hugging Face 下载 DINOv3 骨干 `timm/vit_base_patch16_dinov3.lvd1689m`（约 0.35 GB）。`model.pt` 只存了训练过的权重，冻结的骨干每次都重新加载，所以推理时也要能连上 Hugging Face，或者提前把它缓存到 `HF_HOME`。

### 4.2 数据

按仓库 `setup.md` 的 Step 3–4 操作：

```bash
hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir cs701-sar-course-data
cd cs701-sar-course-data && tar -xzf train.tar.gz && tar -xzf val.tar.gz && cd ..
cd <repo> && python prepare_course_data.py   # 写出 ../dataset/SARFact-Course-20K
```

- `prepare_course_data.py` 用固定种子 0，从 9,392 张有标注图里切出本地 train / val / test：7,512 / 940 / 940 张。
- F2 用 `--train-split train+val+test`，三份都参与训练。官方 val / test 图片没有标注，从不参与训练。
- 两个脚本默认按 `setup.md` 的目录结构找数据。放在别处时，用环境变量指定：

| 变量 | 含义 | Junkai 机器上的值 |
|---|---|---|
| `DATA` | 转换后的数据目录 | `/mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K` |
| `IMAGES` | 排行榜图片目录（val 或 test） | `/mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/val` |

### 4.3 训练（方式 C）

```bash
DATA=/path/to/dataset/SARFact-Course-20K ./train_F2_full.sh
```

脚本执行的就是原始运行的命令（原件见 `run/a3a1_F2_full_ddp.sh`），只把绝对路径换成了变量：

```bash
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node 2 train.py \
  --data "$DATA" --out runs --name F2_full \
  --train-split train+val+test --eval-splits '' --save-every 10 --epochs 100 \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 32 \
  --lora-alpha 64 --train-norms --backbone-lr 3e-4 --compile --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --cascade-size-iou --flip-bias -0.3
```

| 参数 | 作用 |
|---|---|
| `--train-split train+val+test --eval-splits ''` | 用全部有标注数据训练，不留本地评估集 |
| `--epochs 100` | 1 轮 warmup，然后余弦学习率降到 0 |
| `--backbone dinov3 --init pretrained --adapt lora ...` | DINOv3 ViT-B/16 冻结，LoRA r32 / α64 加在所有 block 的四个线性层上 |
| `--train-norms` | 同时训练所有 LayerNorm |
| `--backbone-lr 3e-4` | LoRA 学习率；检测头用默认 `--lr 1e-4` |
| `--cascade --aligned-roi` | 3 阶段 Cascade R-CNN，RoIAlign 对齐到像素中心 |
| `--rpn-nwd 8 --anchor-offset 0.5 --anchor-sizes=...` | RPN 用 NWD 匹配（C = 8），每层 3 个尺寸，最小 6 像素，锚框中心对齐像素 |
| `--cascade-size-iou` | A4：IoU 阈值按目标尺寸放宽（默认表 `a4`） |
| `--flip-bias -0.3` | A2：翻转增强时修正标注的系统偏移 |
| `--compile` | `torch.compile`，加速约 1.3 倍 |
| `NCCL_P2P_DISABLE=1` | 这台机器上 PCIe P2P 会让 NCCL 卡死；别的机器上设了也没有坏处 |

其他参数都是默认值，完整列表在 `run/args.json`：batch 16（两卡各 8），AdamW，weight decay 0.05，bf16，只用翻转增强，seed 0。

- **输出**在 `runs/F2_full/`：`model.pt`（95 MB）、`metrics.json`、`log.txt`、`ckpt_e10.pt` … `ckpt_e90.pt`。
- **核对**：日志开头应是 `parameters: 109,014,752 total, 23,805,152 trainable`。原始运行用了 3.59 小时，每卡显存峰值 38.0 GB，约 36 张/秒；最后几轮的 loss 在 0.05–0.3 之间，可以和 `run/log.txt` 对照。
- **只有 1 张 GPU**：把 `torchrun --standalone --nproc_per_node 2` 换成 `python`，其他不变，约 7 小时，显存峰值约 43 GB。减小 `--batch-size` 也能跑，但结果会和原始运行不同。

### 4.4 生成提交文件

```bash
# 方式 B / C：用权重推理（优先用 weights/，没有就用 runs/F2_full）
IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_F2.sh

# 方式 A：不用 GPU，从保存的原始输出重算后处理
FROM_RAW=1 IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_F2.sh

# Phase 2：换成 test 图片（原始输出只有 val 的，所以只能用方式 B / C）
IMAGES=/path/to/cs701-sar-course-data/test ./make_submission_F2.sh
```

脚本先运行 `predict.py --shift -0.2`，再运行 `tools/check_submission.py` 检查格式。输出文件是 `F2_full_<val|test>_shift-0.2.zip`，用 `OUT=...` 可以改路径。

`predict.py` 的默认后处理就是最终设置（`pp_eval.FINAL`）：
1. 框分数 = 检测概率 × RPN objectness^0.25，然后按类做 Gaussian soft-NMS（σ = 0.75）。
2. 框取三个 Cascade 阶段的平均。
3. 分数阈值 1e-5，每张图至少 30 个框、最多 100 个。
4. 图像类别 = argmax(log p_分类 + log 该类最高检测概率)。
5. 平移：用 `--flip-bias` 训练的模型默认不平移，这里另外指定 `--shift -0.2`（以 512 输入为准）。

检查通过时会输出：

```
OK: 1426 images, 79969 boxes, 0 images without boxes, at most 100 boxes per image and class ..., 0 boxes outside their image, 99.9% of boxes scored > 0.3 have category_id = label + 1
```

### 4.5 核对结果

```bash
sha256sum -c submission/SHA256SUMS   # 原提交文件本身
python - <<'EOF'
import zipfile
a = zipfile.ZipFile('F2_full_val_shift-0.2.zip'); b = zipfile.ZipFile('submission/F2_full_val_shift-0.2.zip')
for n in ('classification.csv', 'detection.json'):
    print(n, a.read(n) == b.read(n))
EOF
```

不要直接比较新旧 zip 的 SHA256：zip 里记录了文件时间，内容相同哈希也不一样。

2026-10-27 的核对结果：
- `code/` 和仓库 commit `51257a2` 逐文件相同。
- 方式 A：两个文件都是 `True`。
- 方式 B：`classification.csv` 为 `True`；`detection.json` 框数相同（79,969），分数最多差 0.0024，是 GPU 上 bf16 计算的误差，对 mAP 的影响可以忽略。
- 方式 C 没有做。

### 4.6 从头训练能复现到什么程度

seed 固定为 0，但 DDP、`torch.compile` 和 cuDNN benchmark 在 GPU 上都不是逐位确定的，重新训练的权重不会和原来完全相同。

波动大小用 C36 测过：37 次运行的本地 val / test mAP 是 37.46±0.35 / 38.91±0.36。100 轮全量训练的波动应该在同一量级，排行榜上约 ±0.5 Δm。

### 4.7 复现本地消融：AA（A4 + A2）对比 C36

F2 相比 F1 的两项改动，是在本地 36 轮实验里验证的：在本地 train（7,512 张）上训练，在本地 val / test（各 940 张）上评估。

```bash
cd code
python train.py --data "$DATA" --out ../runs --name AA_a4a2 --epochs 36 --dump-raw \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 32 --lora-alpha 64 --train-norms --backbone-lr 3e-4 --compile \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
  "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" \
  --cascade-size-iou --flip-bias -0.3
python pp_eval.py --run ../runs/AA_a4a2      # 最终后处理 -> ../runs/AA_a4a2/metrics_pp.json
```

去掉最后一行的 `--cascade-size-iou --flip-bias -0.3` 就是对照组 C36。单卡约 2 小时。

| 实验 | 本地 val mAP | 本地 test mAP |
|---|---|---|
| AA | 38.43 | 40.01 |
| C36（37 次运行） | 37.46±0.35 | 38.91±0.36 |

### 4.8 常见问题

- **`torchrun` 卡在第一步不动**：设置 `NCCL_P2P_DISABLE=1`，脚本里已经设了。
- **显存不足**：单卡 batch 16 需要约 43 GB。改用两卡，或者减小 `--batch-size`（会改变结果）。
- **`predict.py` 加载报错 `_orig_mod`**：说明用的是旧代码；`code/` 里的版本已经处理了 `--compile` 运行保存的权重名。
- **想换检查点**：`predict.py` 加 `--weights ckpt_e90.pt`。
- **没有 GPU**：`predict.py --device cpu`，约 0.7 秒一张。
