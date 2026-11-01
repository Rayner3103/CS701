#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # Number of CPU to request for the job
#SBATCH --mem=24GB                  # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=00-24:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=dinob_lora256_yolo26s_os3_%j.out # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=cs701   # The account you've been assigned (normally student)
#SBATCH --qos=studentqos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=rayner.sim.2024@computing.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=dinob-lora256-yolo26s-os3       # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# DINOv3 ViT-B/16 + LoRA rank 256 (attn.qkv, attn.proj) + YOLO26-s (train_yolo.py), 24 epochs, trained on the
# small-object oversampled data "os3": every train image with a box under 32^2 px (at 512) appears 3 times
# (augment_small_objects.py --oversample 3 --paste-prob 0). Scored on the original local val and test at the end.
# Local reference (RTX 3080, about 4.3 h): val delta_m +15.40, test +15.04 (the same model on the original data: about +9.4 / +10.2).
#
# Submit from the repo (cs701-sar-benchmark):  sbatch gpuClusterRun/dinob_lora256_yolo26s_os3.sh
# Settings, as environment variables (sbatch exports them):
#   EPOCHS=24      number of epochs
#   SEED=0         training seed
#   FRACTION=0.03  use only this fraction of the train images: a few-minute check of the cluster setup (use EPOCHS=1 too)
#   DATA=FILE      the YOLO data.yaml (default: ../dataset/SARFact-YOLO-aug-os3/data_aug.yaml, built here if missing)
#   OFFLINE=1      do not contact Hugging Face: the DINOv3 weights must already be in the cache (see below)
#   EXTRA="..."    more arguments for train_yolo.py, e.g. EXTRA="--batch-size 32" on a bigger GPU
#   e.g.  EPOCHS=1 FRACTION=0.03 sbatch gpuClusterRun/dinob_lora256_yolo26s_os3.sh
#
# Once, on the login node (compute nodes may have no internet), with the venv active:
#   pip install -r requirements-yolo.txt                      # ultralytics on top of the reference environment
#   python -c "import timm; timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True)"   # DINOv3 weights
#   python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"   # COCO weights, saved as yolo26s.pt in the repo folder
#                                                                  # (git ignores yolo*.pt, so a fresh clone lacks it)
# and the data, next to the repo (docs/setup.md): ../dataset/SARFact-Course-20K, then
#   python prepare_yolo_data.py --src ../dataset/SARFact-Course-20K --dst ../dataset/SARFact-YOLO
# The job builds the os3 data itself if it is missing, or build it once yourself:
#   python augment_small_objects.py --oversample 3 --paste-prob 0 --dst ../dataset/SARFact-YOLO-aug-os3
# If the job fails with a connection error, resubmit with OFFLINE=1.
# Resubmitting after a timeout continues from the last epoch (weights/last.pt); a finished run is skipped.

set -euo pipefail

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Python/3.12.8-GCCcore-13.3.0

# This command assumes that you've already created the environment previously
# python3.12 -m venv ~/myenv && source ~/myenv/bin/activate && pip install -r requirements-yolo.txt
source "/common/home/users/r/rayner.sim.2024/repos/CS701/cs701-sar-benchmark/.venv/bin/activate"

# sbatch runs a copy of this script, so the repo is found from the directory the job was submitted from
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_yolo.py ] || { [ -f ../train_yolo.py ] && cd ..; } || { echo "train_yolo.py not found: submit from cs701-sar-benchmark"; exit 1; }
python -c "import torch, timm, ultralytics, pycocotools" 2>/dev/null \
    || { echo "missing Python packages in the venv: pip install -r requirements-yolo.txt"; exit 1; }
[ -f yolo26s.pt ] || { echo "yolo26s.pt not found in $(pwd): see the login-node steps at the top of this script"; exit 1; }
[ -d ../dataset/SARFact-Course-20K/detection ] \
    || { echo "no ../dataset/SARFact-Course-20K (train_yolo.py scores on it): see docs/setup.md"; exit 1; }

# The os3 data: built once (about 30 s, symlinks and labels only); a finished build has its data_aug.yaml
DATA="${DATA:-$(pwd)/../dataset/SARFact-YOLO-aug-os3/data_aug.yaml}"
if [ ! -f "$DATA" ]; then
    [ -d ../dataset/SARFact-YOLO/train ] \
        || { echo "no ../dataset/SARFact-YOLO: python prepare_yolo_data.py --src ../dataset/SARFact-Course-20K --dst ../dataset/SARFact-YOLO"; exit 1; }
    python augment_small_objects.py --oversample 3 --paste-prob 0 --dst ../dataset/SARFact-YOLO-aug-os3
    [ -f "$DATA" ] || { echo "$DATA was not built (DATA points elsewhere than the os3 folder?)"; exit 1; }
fi

[ "${OFFLINE:-0}" = "1" ] && export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less memory fragmentation
EPOCHS="${EPOCHS:-24}"
SEED="${SEED:-0}"
NAME="aug_os3_dinob_lora256_yolo26s_ep${EPOCHS}${FRACTION:+_fraction$FRACTION}${SEED:+_seed$SEED}"
OUT="runs/yolo"   # one fixed folder per setting, so a resubmitted job continues from weights/last.pt
RUN="$OUT/$NAME"
mkdir -p "$OUT"

echo "job ${SLURM_JOB_ID:-local} on $(hostname), started $(date '+%F %T')"
echo "repo $(pwd), data $DATA, run $RUN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'"

if [ -f "$RUN/metrics.json" ]; then
    echo "$RUN is finished (metrics.json): nothing to do"
    exit 0
fi

# 24 epochs, AdamW lr 1e-3 (the DINO LoRA weights at lr x 0.1), batch size 16, gradient checkpointing in the ViT: as the
# local run, where LoRA on a ViT-B needs it to fit a 10 GB GPU (on a bigger GPU it costs time, not accuracy).
# The best epoch (best.pt) is scored on the local val and test splits at the end, into $RUN/metrics.json.
# shellcheck disable=SC2086  # EXTRA is meant to split into arguments
if [ -f "$RUN/weights/last.pt" ]; then
    echo "resuming $RUN from weights/last.pt"
    srun --gres=gpu:1 python train_yolo.py --yolo yolo26s --dino vit_base_patch16_dinov3.lvd1689m --dino-adapt lora \
        --lora-rank 256 --dino-grad-ckpt --epochs "$EPOCHS" --seed "$SEED" --workers "${SLURM_CPUS_PER_TASK:-4}" \
        --data "$DATA" --out "$OUT" --name "$NAME" ${FRACTION:+--fraction $FRACTION} ${EXTRA:-} --resume
else
    srun --gres=gpu:1 python train_yolo.py --yolo yolo26s --dino vit_base_patch16_dinov3.lvd1689m --dino-adapt lora \
        --lora-rank 256 --dino-grad-ckpt --epochs "$EPOCHS" --seed "$SEED" --workers "${SLURM_CPUS_PER_TASK:-4}" \
        --data "$DATA" --out "$OUT" --name "$NAME" ${FRACTION:+--fraction $FRACTION} ${EXTRA:-}
fi

echo "finished $(date '+%F %T'); results in $RUN (metrics.json, weights/best.pt)"
python -c "
import json, sys
m = json.load(open('$RUN/metrics.json'))
for s in ('val', 'test'):
    print(s, {k: round(m[s][k], 3) for k in ('accuracy', 'macro_f1', 'mAP', 'AP50', 'delta_m')})
"
