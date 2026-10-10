# Setup Guide

How to go from nothing to a working smoke test, and how to share your work with the team.

The scripts find the data by its position next to the repo, so keep this layout (the top folder can have any
name; `CS701/` is used below):

```
CS701/
├── cs701-sar-benchmark/        # this repo (Step 1)
├── cs701-sar-course-data/      # raw download from Hugging Face (Step 3)
└── dataset/SARFact-Course-20K/ # converted data that train.py reads (Step 4)
```

# Step 1: Clone the repo

```bash
mkdir CS701 && cd CS701
git clone https://github.com/Rayner3103/CS701.git cs701-sar-benchmark
cd cs701-sar-benchmark
git remote add upstream https://github.com/doem97/cs701-sar-benchmark   # the TA's original code
```

Set up the Python environment (once per machine):

```bash
conda create -n cs701bench python=3.12 -y
conda activate cs701bench
pip install -r requirements.txt   # torch 2.9.0 / torchvision 0.24.0 built for CUDA 12.8
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: 2.9.0+cu128 True
```

Run `conda activate cs701bench` in every new terminal before the steps below.

# Step 2: Create and work on the branch with <name>

Each member works on their own branch, named after themselves. Never commit straight to `main`.

```bash
git checkout -b <name>        # first time: create your branch
git checkout <name>           # later: switch back to it
git branch                    # the * marks the branch you are on
```

# Step 3: Download the dataset

Run this from the top folder (`CS701/`), not inside the repo:

```bash
cd ..                                   # from cs701-sar-benchmark/ to CS701/
pip install -U huggingface_hub
hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir cs701-sar-course-data
cd cs701-sar-course-data && tar -xzf train.tar.gz && cd ..
```

You should now have `cs701-sar-course-data/train/` with `images/` (9,392 images), `labels.csv` and
`instances.json`.

New splits are released later: val on 12 Oct 2026 and test on 2 Nov 2026. After each release, run the `hf download`
command again and unpack the new archive (`tar -xzf val.tar.gz`, `tar -xzf test.tar.gz`).

# Step 4: Prepare the dataset

`train.py` expects a different folder layout from the Hugging Face download, and the released val/test have no
labels. `prepare_course_data.py` converts the download and holds out part of train for local val/test:

```bash
cd cs701-sar-benchmark
python prepare_course_data.py
```

Expected output:

```
train: 7512 images, 26015 boxes
val: 940 images, 3323 boxes
test: 940 images, 3195 boxes
```

- It writes `../dataset/SARFact-Course-20K/` with `classification/{train,val,test}.csv` and
  `detection/instances_{train,val,test}.json`.
- Images are not copied: `dataset/SARFact-Course-20K/train` is a symlink to `cs701-sar-course-data/train`.
- The split is stratified by class with a fixed seed (`--seed 0`), so every member gets the same split and results
  can be compared. Change the sizes with `--val-frac` / `--test-frac` (default 0.1 each).
- The "val" and "test" here are local hold-outs from the labelled train set, not the Codabench splits.

# Step 5: Run smoke test

A one-epoch run on 64 images that checks data loading, the model, training and evaluation end to end. The ViT
weights (0.4 GB) download from Hugging Face on the first run.

```bash
python train.py --backbone vit --init pretrained --adapt lora --limit 64 --epochs 1 --batch-size 4 --workers 4 --out runs/smoke
```

It works if the log ends with lines like these (it takes a minute or two):

```
parameters: 107,039,257 total, 20,997,913 trainable
epoch 1 iter 16/16 loss 2.628 (...)
epoch 1 val: accuracy ... macro_f1 ... mAP 0.0000 AP50 0.0000
test: accuracy ... macro_f1 ... mAP 0.0000 AP50 0.0000
```

- mAP 0 is normal here: one epoch on 64 images is too little to learn boxes. The smoke test only checks that
  everything runs.
- Results go to `runs/smoke/vit_pretrained_lora/` (`log.txt`, `metrics.json`, predictions, `model.pt`).
- With `--spatial-prior` (the small CNN branch for small objects) the folder is `vit_pretrained_lora_sp/`.
- Optional: `python -m pytest tests` runs the code checks. Three `test_data` tests fail by design, because they
  expect the TA's split sizes, not the split from Step 4. The full fine-tuning overfit test needs more than 10 GB of
  GPU memory.

A full training run (24 epochs) drops `--limit` and `--epochs`:

```bash
python train.py --backbone vit --init pretrained --adapt lora --batch-size 8
```

The default batch size of 16 was set for a 96 GB GPU. On a 10-12 GB GPU use `--batch-size 8`, or 4 if you get
`CUDA out of memory`.

# Step 6: Sync up with git

Commit your work to your own branch and push it:

```bash
git status                     # see what changed
git add <files>                # add code only: not data, runs/ or model.pt
git commit -m "Short description of the change"
git push -u origin <name>      # -u only the first time; afterwards just: git push
```

Then open a pull request from `<name>` into `main` on GitHub so the team can review and merge it.

Get the team's latest `main` into your branch:

```bash
git fetch origin
git merge origin/main          # run while on your branch; fix any conflicts, then commit
```

Data and results never go into git: the data lives outside the repo, and `runs/` is in `.gitignore`. Share
results by posting the numbers from `metrics.json`.
