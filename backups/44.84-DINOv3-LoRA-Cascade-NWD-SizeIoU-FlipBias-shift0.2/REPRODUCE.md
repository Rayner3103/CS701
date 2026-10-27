# 复现 F2_full_val_shift-0.2.zip

这份文档说明怎样从零开始重新得到 `submission/F2_full_val_shift-0.2.zip`，包括准备环境和数据、训练、生成提交文件和检查结果。另外也说明怎样在本地复现支撑 F2 的消融实验（AA 对比 C36）。

文中的命令都在本文件夹（`backups/44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/`）里运行。训练和推理用的是 `code/` 里的代码快照，它和仓库 commit `51257a2` 逐文件相同。

## 0. 三种复现方式

| 方式 | 需要什么 | 耗时 | 结果 |
|---|---|---|---|
| A. 只重算后处理 | 本机 `weights/F2_full_val_raw.pt`，不用 GPU | 几秒 | zip 里的两个文件和原提交逐字节相同 |
| B. 用训练好的权重推理 | 本机 `weights/model.pt`，1 张 GPU | 约 1 分钟 | 分类相同，框数相同，框分数差 ≤0.003 |
| C. 从头训练 | 2 张约 48 GB 的 GPU | 约 3.6 小时 + 推理 | 随机波动范围内一致（见第 6 节） |

`weights/` 没有放进 git，只在 Junkai 的机器上：`/mnt/hdd/junkaichen/CS701/CS701/backups/44.84-DINOv3-LoRA-Cascade-NWD-SizeIoU-FlipBias-shift0.2/weights/`。别的机器上只能用方式 C。

## 1. 环境

```bash
conda create -n cs701bench python=3.12 -y
conda activate cs701bench
pip install -r code/requirements.txt   # torch 2.9.0+cu128, torchvision 0.24.0, timm 1.0.24, pycocotools 2.0.11
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.9.0+cu128 True
```

- 当时环境的完整包列表在 `run/pip_freeze.txt`。
- 原训练用的硬件：2 × NVIDIA RTX 6000 Ada（48 GB），CUDA 12.8。
- 第一次运行会从 Hugging Face 下载 DINOv3 骨干 `timm/vit_base_patch16_dinov3.lvd1689m`（约 0.35 GB）。`model.pt` 只存了训练过的那部分权重，冻结的骨干每次都从这里重新加载，所以推理时也要能连上 Hugging Face，或者提前把它下载到 `HF_HOME` 缓存里。

## 2. 数据

按仓库 `setup.md` 的 Step 3–4 操作：

```bash
hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir cs701-sar-course-data
cd cs701-sar-course-data && tar -xzf train.tar.gz && tar -xzf val.tar.gz && cd ..
cd <repo> && python prepare_course_data.py   # 写出 ../dataset/SARFact-Course-20K
```

`prepare_course_data.py` 用固定种子（seed 0）从 9,392 张有标注图里切出本地 train / val / test，分别是 7,512 / 940 / 940 张。F2 用 `--train-split train+val+test`，所以三份都参与训练，也就是全部 9,392 张有标注图。Codabench 的 val / test 图片没有标注，从来不参与训练。

两个脚本默认按 `setup.md` 的目录结构找数据（仓库旁边的 `dataset/` 和 `cs701-sar-course-data/`）。数据放在别处时，用两个环境变量指定：

| 变量 | 含义 | Junkai 机器上的值 |
|---|---|---|
| `DATA` | 转换后的数据目录 | `/mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K` |
| `IMAGES` | Codabench 图片目录（val 或 test） | `/mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/val` |

## 3. 训练（方式 C）

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

各参数的含义：

| 参数 | 作用 |
|---|---|
| `--train-split train+val+test --eval-splits ''` | 用全部有标注数据训练，不留本地评估集 |
| `--epochs 100` | 1 轮 warmup，然后余弦学习率降到 0 |
| `--backbone dinov3 --init pretrained --adapt lora ...` | DINOv3 ViT-B/16 冻结，LoRA r32 / α64 加在所有 block 的四个线性层上 |
| `--train-norms` | 同时训练所有 LayerNorm |
| `--backbone-lr 3e-4` | LoRA 学习率；检测头用默认 `--lr 1e-4` |
| `--cascade --aligned-roi` | 3 阶段 Cascade R-CNN，RoIAlign 对齐到像素中心 |
| `--rpn-nwd 8 --anchor-offset 0.5 --anchor-sizes=...` | RPN 用 NWD 匹配（C = 8），每层 3 个尺寸，最小 6 像素，锚框中心对齐像素 |
| `--cascade-size-iou` | A4：三个阶段的 IoU 阈值按目标尺寸放宽（默认表 `a4`） |
| `--flip-bias -0.3` | A2：翻转增强时修正标注的系统偏移 |
| `--compile` | `torch.compile`，加速约 1.3 倍 |
| `NCCL_P2P_DISABLE=1` | 这台机器上 PCIe P2P 会让 NCCL 卡死；别的机器上设了也没有坏处 |

其他参数都是默认值，完整列表在 `run/args.json`：batch 16（两卡各 8），AdamW，weight decay 0.05，bf16，只用翻转增强，seed 0。

**输出**在 `runs/F2_full/`：
- `model.pt`：训练过的权重，95 MB
- `metrics.json`：参数和参数量
- `log.txt`：训练日志
- `ckpt_e10.pt` … `ckpt_e90.pt`：中途的检查点

**检查点**：
- 日志开头应该是 `parameters: 109,014,752 total, 23,805,152 trainable`，和 `run/log.txt` 一致。
- 原始运行用了 3.59 小时，每卡显存峰值 38.0 GB，速度约 36 张/秒。
- 最后几轮的 loss 应在 0.05–0.3 之间，可以和 `run/log.txt` 对照。

**只有 1 张 GPU 时**，把 `torchrun --standalone --nproc_per_node 2` 换成 `python`，其他参数不变，约 7 小时，显存峰值约 43 GB。显存不够时可以减小 `--batch-size`，但结果就和原始运行不同了。

## 4. 生成提交文件

```bash
# 方式 B / C：用权重推理（优先用 weights/，没有就用 runs/F2_full）
IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_F2.sh

# 方式 A：不用 GPU，从保存的原始输出重算后处理
FROM_RAW=1 IMAGES=/path/to/cs701-sar-course-data/val ./make_submission_F2.sh

# Phase 2：换成 test 图片（原始输出只有 val 的，所以只能用方式 B / C）
IMAGES=/path/to/cs701-sar-course-data/test ./make_submission_F2.sh
```

脚本先运行 `predict.py --shift -0.2`，再运行 `tools/check_submission.py` 检查格式。输出文件是 `F2_full_<val|test>_shift-0.2.zip`，用 `OUT=...` 可以改路径。

`predict.py` 的默认后处理就是最终设置（`pp_eval.FINAL`），所以不需要额外参数：
1. 框分数 = 检测概率 × RPN objectness^0.25，然后按类做 Gaussian soft-NMS（σ = 0.75）。
2. 输出的框取三个 Cascade 阶段的框的平均。
3. 分数阈值 1e-5，每张图至少保留 30 个框、最多 100 个。
4. 图像类别 = argmax(log p_分类 + log 该类最高检测概率)。
5. 平移：用 `--flip-bias` 训练的模型默认不平移。这里另外指定了 `--shift -0.2`，把所有框平移 −0.2 像素（以 512 输入为准）。

检查通过时会输出：

```
OK: 1426 images, 79969 boxes, 0 images without boxes, at most 100 boxes per image and class ..., 0 boxes outside their image, 99.9% of boxes scored > 0.3 have category_id = label + 1
```

## 5. 核对结果

```bash
sha256sum -c submission/SHA256SUMS   # 原提交文件本身
python - <<'EOF'
import zipfile
a = zipfile.ZipFile('F2_full_val_shift-0.2.zip'); b = zipfile.ZipFile('submission/F2_full_val_shift-0.2.zip')
for n in ('classification.csv', 'detection.json'):
    print(n, a.read(n) == b.read(n))
EOF
```

不要直接比较新旧 zip 文件的 SHA256。zip 里记录了文件时间，即使内容完全相同，哈希也不一样。

2026-10-27 的核对结果：
- 方式 A：两个文件都是 `True`。
- 方式 B：`classification.csv` 为 `True`，`detection.json` 框数相同（79,969），但分数有最多 0.0024 的差别。这是 GPU 上 bf16 计算的误差，对 mAP 的影响可以忽略。

## 6. 从头训练能复现到什么程度

seed 固定为 0，但 DDP、`torch.compile` 和 cuDNN benchmark 在 GPU 上都不是逐位确定的，所以重新训练得到的权重和原来的不会完全相同，分数会有随机波动。

波动的大小用同一配方的 36 轮版本（C36）测过：36 个不同随机种子的本地 val / test mAP 是 37.46±0.35 / 38.90±0.37。100 轮全量训练的波动应该在同一量级，Codabench 上约 ±0.5 Δm。

## 7. 复现本地消融：AA（A4 + A2）对比 C36

F2 相比 F1 的两项改动是在本地 36 轮实验里验证的。在本地 train（7,512 张）上训练，在本地 val / test（各 940 张）上评估：

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

- 去掉最后一行的 `--cascade-size-iou --flip-bias -0.3`，就是对照组 C36。
- 单卡约 2 小时。
- 原始结果（最终后处理，不平移）：

| 实验 | 本地 val mAP | 本地 test mAP |
|---|---|---|
| AA | 38.43 | 40.01 |
| C36（36 个种子的平均） | 37.46±0.35 | 38.90±0.37 |

## 8. 常见问题

- **`torchrun` 卡在第一步不动**：设置 `NCCL_P2P_DISABLE=1`，脚本里已经设了。
- **显存不足**：单卡 batch 16 需要约 43 GB。可以改用两卡，或者减小 `--batch-size`，后者会改变结果。
- **`predict.py` 加载报错 `_orig_mod`**：说明用的是旧代码。`code/` 里的版本已经处理了 `--compile` 运行保存的权重名。
- **想换检查点**：`predict.py` 加 `--weights ckpt_e90.pt`。
- **没有 GPU**：`predict.py --device cpu` 可以跑，约 0.7 秒一张。
