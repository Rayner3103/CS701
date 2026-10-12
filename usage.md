# Using run_all_3080.sh

`run_all_3080.sh` trains the four pretrained adapter configurations twice each, without and with the spatial prior
(the small CNN branch for small objects), at batch size 8 so they fit a 10 GB RTX 3080. Use it to measure whether
the spatial prior helps.

Before the first run, finish Steps 1-4 of [setup.md](setup.md) (clone, environment, download and prepare the data),
and run `conda activate cs701bench`. Run everything below from `cs701-sar-benchmark/`.

## What it runs

Eight runs, in this order. Each pair runs back to back, so stopping the script early still leaves complete pairs.

| # | Backbone | Adaptation | Spatial prior | Results folder (under `OUT_FOLDER`) |
|---|---|---|---|---|
| 1 | ViT | LoRA | no | `vit_pretrained_lora` |
| 2 | ViT | LoRA | yes | `vit_pretrained_lora_sp` |
| 3 | ViT | MoE-LoRA | no | `vit_pretrained_moelora` |
| 4 | ViT | MoE-LoRA | yes | `vit_pretrained_moelora_sp` |
| 5 | TerraMind | LoRA | no | `terramind_pretrained_lora` |
| 6 | TerraMind | LoRA | yes | `terramind_pretrained_lora_sp` |
| 7 | TerraMind | MoE-LoRA | no | `terramind_pretrained_moelora` |
| 8 | TerraMind | MoE-LoRA | yes | `terramind_pretrained_moelora_sp` |

All eight are within the challenge limits (about 21 M trainable, 107 M total parameters).

## Commands

```bash
[SEED=N] [OUT_FOLDER=DIR] bash run_all_3080.sh [EPOCHS] [extra train.py arguments]
```

- `EPOCHS`: number of training epochs for every run (default 24).
- `SEED`: the random seed of every run (default 0). All eight runs use the same seed, and the spatial prior does not
  change the random state, so the two runs of a pair start from the same weights and see the same data order and
  flips: the only difference is the spatial prior. (Runs are still not bit-identical: some GPU operations are
  nondeterministic.)
- `OUT_FOLDER`: the folder for everything this launch writes (default `runs`): the eight run folders, the combined
  log and the launch details (see [Outputs](#outputs)). Use it instead of `--out`.
- Anything after `EPOCHS` is passed to every `train.py` call (see `python train.py -h`).

| Goal | Command |
|---|---|
| Quick check that all eight runs start (a few minutes) | `OUT_FOLDER=runs/smoke bash run_all_3080.sh 1 --limit 16` |
| Short comparison | `bash run_all_3080.sh 12` |
| Full runs, as in the README, in their own folder | `OUT_FOLDER=runs/full_seed0 bash run_all_3080.sh 24` |
| Pick the best epoch more finely | `bash run_all_3080.sh 24 --eval-every 2` |
| Another seed, in its own folder | `SEED=1 OUT_FOLDER=runs/full_seed1 bash run_all_3080.sh 24` |

Use a different `OUT_FOLDER` for every set of runs you want to keep. Launching again into the same folder with the
same `EPOCHS` resumes it (see [Resuming](#resuming-after-a-crash-or-a-stop)); with a different `EPOCHS`, its runs
start over.

## Resuming after a crash or a stop

Every run saves `last_checkpoint.pt` after each completed epoch: weights, optimizer, learning-rate schedule, random
state and val history. To continue, launch the same command again (same `OUT_FOLDER` and `EPOCHS`):

```bash
OUT_FOLDER=runs/full_seed0 bash run_all_3080.sh 24
```

For each of the eight runs, the script reads its `log.txt` and prints what it does:

| The run's `log.txt` | What happens | Printed |
|---|---|---|
| ends with the final results of epoch `EPOCHS` (`test (last, epoch 24)`) | complete: skipped | `finished all 24 epochs: skipped` |
| stops in epoch n, and `last_checkpoint.pt` exists | resumes from the last completed epoch (n - 1 when it crashed during epoch n) and continues its `log.txt` | `log.txt stops in epoch n: resuming from the last completed epoch` |
| stops in epoch n, no checkpoint (it crashed in epoch 1, or it was trained before checkpoints existed) | starts over | `... there is no checkpoint: starting over` |
| is from a run of a different number of epochs | starts over (the learning-rate schedule depends on the total) | `... not 24: starting over` |
| does not exist | starts | `no log.txt yet: starting` |

- The resumed run's `log.txt` shows `resumed from the checkpoint of epoch k` where it continued.
- The script's own `run_all_3080.log` and `run_all_3080_details.txt` are appended to, so they keep every launch.
- A resumed run ends with the same files as an uninterrupted one, and `last_checkpoint.pt` is deleted once the run
  is complete. Its results can differ slightly from an uninterrupted run: the augmentation flips after the resume
  are drawn differently.
- A single run can also be resumed by hand: add `--resume` to its `train.py` command.

## Before a long run: hold the NVIDIA driver

Ubuntu installs updates automatically (`unattended-upgrades`). If it upgrades the NVIDIA driver while you train, the
driver files on disk no longer match the driver loaded in memory until you reboot. Runs started after the upgrade
log `Can't initialize NVML` and can crash with `NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT
FAILED` (this happened on 11 Oct 2026: the driver went from 595.91.07 to 595.99.02 in the middle of the runs).

1. Check that the driver works, and reboot first if it does not:

   ```bash
   nvidia-smi    # "Failed to initialize NVML: Driver/library version mismatch" means: reboot first
   ```

2. Hold all NVIDIA packages, so updates skip them while you train:

   ```bash
   sudo apt-mark hold $(dpkg-query -W -f='${binary:Package}\n' | grep -E '^(nvidia-|libnvidia-|xserver-xorg-video-nvidia-)')
   apt-mark showhold    # lists the held packages
   ```

3. When training is done, unhold them, update, and reboot, so the driver stays current and secure:

   ```bash
   sudo apt-mark unhold $(apt-mark showhold | grep -E '^(nvidia-|libnvidia-|xserver-xorg-video-nvidia-)')
   sudo apt update && sudo apt upgrade
   sudo reboot
   ```

A hold lasts until you unhold, across reboots. Do not leave the driver held for weeks: you would miss security fixes.

## Running it overnight

Full runs take many hours, so start them in the background so they keep going after you close the terminal:

```bash
OUT_FOLDER=runs/full_seed0 nohup bash run_all_3080.sh 24 > /dev/null 2>&1 &
```

The script writes its own log to `OUT_FOLDER/run_all_3080.log`, so nohup's output can be discarded. Check on it:

```bash
grep -E "^===|val:|test|Failed|finished" runs/full_seed0/run_all_3080.log   # which run, and its results so far
tail -f runs/full_seed0/run_all_3080.log     # live log (Ctrl+C stops watching, not training)
nvidia-smi                                   # GPU memory and use
```

Stop it: `pkill -f run_all_3080.sh && pkill -f train.py`.

## Outputs

`OUT_FOLDER` holds everything from one launch:

```
OUT_FOLDER/
├── run_all_3080.log            # everything the script printed: each command, all losses, results, failed runs
├── run_all_3080_details.txt    # the launch: start/finish time, command, epochs, seed, extra arguments,
│                               #   git branch and commit (and whether it had uncommitted changes),
│                               #   Python, torch version, GPU and driver version
├── vit_pretrained_lora/        # one folder per run (see the table above)
└── ...
```

Each run folder holds:

| File | Contents |
|---|---|
| `log.txt` | settings, losses, val results, test results |
| `metrics.json` | parameter counts, time, memory, all val results, test results, and the best epoch's results (`best`) |
| `best_model.pt` | the model with the best val Δm, saved as soon as it improves (409 MB) |
| `model.pt` | the last-epoch model (409 MB) |
| `predictions_*.json` | val and test detections of the last and the best model |
| `last_checkpoint.pt` | everything needed to resume, after the last completed epoch (about 600 MB); only while the run is not complete |

- "Best" means the highest val Δm (the leaderboard metric, from macro-F1 and mAP). Validation runs every 4 epochs
  by default, so the best epoch is one of 4, 8, ..., 24 unless you pass `--eval-every`.
- The eight runs need about 6.5 GB of disk for the checkpoints.

## Comparing the results

```bash
python compare_runs.py runs/full_seed0     # the OUT_FOLDER of the launch
```

It prints one line per run with test macro-F1, mAP and Δm for the last epoch, plus the best epoch and its test Δm.
Compare each `_sp` run with the run above it: the spatial prior targets small objects, so look at mAP first.

- Δm here uses the README's val reference values. Your val and test splits are held out from the course train set,
  so the numbers are only for comparing your own runs, not with the README's table or the leaderboard.
- Small differences (under about 1 Δm) can be noise between seeds. Before concluding, repeat the pair that matters
  with another `--seed`.

## When something goes wrong

- A failed run does not stop the script: the next run starts, and the failed ones are listed at the end of the log.
  Fix the cause, then launch the same command again to resume them.
- `CUDA out of memory`: TerraMind + MoE-LoRA is the tightest (8.7 of 9.6 GB). Close other programs using the GPU,
  or rerun just that configuration with a smaller batch:
  `python train.py --backbone terramind --init pretrained --adapt moelora --batch-size 4 [--spatial-prior]`
  (a different batch size changes training, so note it when comparing).
- `Can't initialize NVML` in the log, or a crash with `nvmlInit_v2_() INTERNAL ASSERT FAILED`: the NVIDIA driver
  was upgraded during training. Reboot, rerun the failed runs, and hold the driver next time (see
  [Before a long run](#before-a-long-run-hold-the-nvidia-driver)).
- The script sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself; without it TerraMind + MoE-LoRA runs
  out of memory at batch size 8.
- Data errors (file not found): redo Step 4 of [setup.md](setup.md).
