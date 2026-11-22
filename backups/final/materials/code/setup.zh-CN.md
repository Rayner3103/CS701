# 环境配置指南

[English](setup.md) · **中文**

如何从零开始跑通烟测，以及如何把你的工作分享给团队。

脚本根据数据相对仓库的位置来找数据，所以请保持下面的布局（顶层文件夹可以取任意名字；下面用 `CS701/`）：

```
CS701/
├── cs701-sar-benchmark/        # this repo (Step 1)
├── cs701-sar-course-data/      # raw download from Hugging Face (Step 3)
└── dataset/SARFact-Course-20K/ # converted data that train.py reads (Step 4)
```

# 步骤 1：克隆仓库

```bash
mkdir CS701 && cd CS701
git clone https://github.com/Rayner3103/CS701.git cs701-sar-benchmark
cd cs701-sar-benchmark
```

配置 Python 环境（每台机器一次）：

```bash
conda create -n cs701bench python=3.12 -y
conda activate cs701bench
pip install -r requirements.txt   # torch 2.9.0 / torchvision 0.24.0 built for CUDA 12.8
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: 2.9.0+cu128 True
```

在执行下面的步骤之前，每个新终端都要先运行 `conda activate cs701bench`。

# 步骤 2：创建并使用 <name> 分支

每位成员在自己的分支上工作，分支以自己的名字命名。绝不要直接提交到 `main`。

```bash
git checkout -b <name>        # first time: create your branch
git checkout <name>           # later: switch back to it
git branch                    # the * marks the branch you are on
```

# 步骤 3：下载数据集

在顶层文件夹（`CS701/`）运行，不要在仓库内运行：

```bash
cd ..                                   # from cs701-sar-benchmark/ to CS701/
pip install -U huggingface_hub
hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir cs701-sar-course-data
cd cs701-sar-course-data && tar -xzf train.tar.gz && cd ..
```

现在你应该有 `cs701-sar-course-data/train/`，其中包含 `images/`（9,392 张图像）、`labels.csv` 和 `instances.json`。

新的划分之后发布：val 在 2026 年 10 月 12 日，test 在 2026 年 11 月 2 日。每次发布后，重新运行 `hf download` 命令并解压新的压缩包（`tar -xzf val.tar.gz`、`tar -xzf test.tar.gz`）。

# 步骤 4：准备数据集

`train.py` 需要的文件夹布局与 Hugging Face 下载的不同，而且发布的 val/test 没有标签。`prepare_course_data.py` 负责转换下载内容，并从 train 中留出一部分作为本地 val/test：

```bash
cd cs701-sar-benchmark
python prepare_course_data.py
```

预期输出：

```
train: 7512 images, 26015 boxes
val: 940 images, 3323 boxes
test: 940 images, 3195 boxes
```

- 它写出 `../dataset/SARFact-Course-20K/`，其中包含 `classification/{train,val,test}.csv` 和 `detection/instances_{train,val,test}.json`。
- 图像不会被复制：`dataset/SARFact-Course-20K/train` 是指向 `cs701-sar-course-data/train` 的符号链接。
- 划分按类别分层，使用固定 seed（`--seed 0`），因此每位成员得到相同的划分，结果可以相互比较。用 `--val-frac` / `--test-frac` 修改大小（默认各 0.1）。
- 这里的 “val” 和 “test” 是从有标签 train 集中划出的本地留出集，不是 Codabench 的划分。

# 步骤 5：运行烟测

在 64 张图像上运行一轮，端到端检查数据加载、模型、训练和评估。ViT 权重（0.4 GB）在首次运行时从 Hugging Face 下载。

```bash
python train.py --backbone vit --init pretrained --adapt lora --limit 64 --epochs 1 --batch-size 4 --workers 4 --out runs/smoke
```

如果日志以类似下面的行结尾，就说明能正常工作（需要一两分钟）：

```
parameters: 107,039,257 total, 20,997,913 trainable
epoch 1 iter 16/16 loss 2.628 (...)
epoch 1 val: accuracy ... macro_f1 ... mAP 0.0000 AP50 0.0000
test: accuracy ... macro_f1 ... mAP 0.0000 AP50 0.0000
```

- 这里 mAP 为 0 是正常的：在 64 张图像上训练一轮太少，学不到框。烟测只检查一切能否运行。
- 结果写入 `runs/smoke/vit_pretrained_lora/`（`log.txt`、`metrics.json`、预测、`model.pt`）。
- 使用 `--spatial-prior`（针对小目标的小型 CNN 分支）时，文件夹为 `vit_pretrained_lora_sp/`。
- 可选：`python -m pytest tests` 运行代码检查。有三个 `test_data` 测试按设计会失败，因为它们预期的是助教的划分大小，而不是步骤 4 的划分。完全微调的过拟合测试需要超过 10 GB 的 GPU 内存。

完整的训练运行（24 轮）去掉 `--limit` 和 `--epochs`：

```bash
python train.py --backbone vit --init pretrained --adapt lora --batch-size 8
```

默认 batch size 16 是为 96 GB GPU 设定的。在 10-12 GB 的 GPU 上使用 `--batch-size 8`，如果出现 `CUDA out of memory` 则用 4。

# 步骤 6：用 git 同步

把你的工作提交到自己的分支并推送：

```bash
git status                     # see what changed
git add <files>                # add code only: not data, runs/ or model.pt
git commit -m "Short description of the change"
git push -u origin <name>      # -u only the first time; afterwards just: git push
```

然后在 GitHub 上从 `<name>` 向 `main` 发起 pull request，以便团队审查和合并。

把团队最新的 `main` 合并到你的分支：

```bash
git fetch origin
git merge origin/main          # run while on your branch; fix any conflicts, then commit
```

数据和结果绝不进入 git：数据放在仓库之外，`runs/` 在 `.gitignore` 中。分享结果时，发布 `metrics.json` 中的数字即可。
