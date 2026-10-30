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
#SBATCH --time=01:00:00             # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=smoke_%j.out       # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=cs701   # The account you've been assigned (normally student)
#SBATCH --qos=studentqos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=rayner.sim.2024@computing.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=cs701-smoke     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Smoke test of the current best model: DINOv3 ViT-B/16 + LoRA on attention and MLP + 3-stage Cascade R-CNN head
# (with the stage heads checkpointed: the same results for ~2 GB less GPU memory), batch size 8. It trains 1 epoch,
# validates, tests, and saves a checkpoint: the whole pipeline, to check the cluster setup and to time one epoch and
# read the peak GPU memory (both are in the last lines of the log) before submitting the real 24-epoch run.
#
# Submit from the repo (cs701-sar-benchmark):  sbatch gpuClusterRun/smokeTest.sh
# Settings, as environment variables (sbatch exports them):
#   LIMIT=256   only the first 256 images of every split: a ~2 minute check instead of a ~15 minute full epoch
#   DATA=DIR    the prepared data (default: ../dataset/SARFact-Course-20K, where prepare_course_data.py writes it)
#   VENV=DIR    the virtual environment (default: ~/myenv)
#   OFFLINE=1   do not contact Hugging Face: the weights must already be in the cache (see below)
#   EXTRA="..." more arguments for train.py, e.g. EXTRA="--batch-size 16" on a bigger GPU
#   e.g.  LIMIT=256 sbatch gpuClusterRun/smokeTest.sh
#
# Once, on the login node (compute nodes may have no internet), with the venv active and
# `pip install -r requirements.txt` done: download the DINOv3 weights (0.4 GB) into the Hugging Face cache, which
# lives in your home directory and so is shared with the compute nodes:
#   python -c "import timm; timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True)"
# If the job then fails with a connection error, resubmit with OFFLINE=1.

set -euo pipefail

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Python/3.12.8-GCCcore-13.3.0

# This command assumes that you've already created the environment previously
# python3.12 -m venv ~/myenv && source ~/myenv/bin/activate && pip install -r requirements.txt
source "/common/home/users/r/rayner.sim.2024/repos/CS701/cs701-sar-benchmark/.venv/bin/activate"

# sbatch runs a copy of this script, so the repo is found from the directory the job was submitted from
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train.py ] || { [ -f ../train.py ] && cd ..; } || { echo "train.py not found: submit from cs701-sar-benchmark"; exit 1; }
DATA="${DATA:-$(pwd)/../dataset/SARFact-Course-20K}"
[ -d "$DATA/train" ] || { echo "no prepared data in $DATA: run prepare_course_data.py or set DATA=..."; exit 1; }
python -c "import torch, timm, terratorch, pycocotools" 2>/dev/null \
    || { echo "missing Python packages in the venv: pip install -r requirements.txt"; exit 1; }

[ "${OFFLINE:-0}" = "1" ] && export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less memory fragmentation
OUT="runs/smoke_cluster/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

echo "job ${SLURM_JOB_ID:-local} on $(hostname), started $(date '+%F %T')"
echo "repo $(pwd), data $DATA, output $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'"

# Submit your job to the cluster
# shellcheck disable=SC2086  # EXTRA is meant to split into arguments
srun --gres=gpu:1 python train.py --backbone dinov3 --init pretrained --adapt lora --lora-targets attn+mlp \
    --cascade --cascade-checkpoint --batch-size 8 --epochs 1 --eval-every 1 --seed 0 \
    --workers "${SLURM_CPUS_PER_TASK:-4}" --data "$DATA" --out "$OUT" ${LIMIT:+--limit $LIMIT} ${EXTRA:-}

echo "finished $(date '+%F %T'); results in $OUT/dinov3_pretrained_lora_mlp_cascade"
grep -E "^parameters|^test \(|^training" "$OUT"/dinov3_pretrained_lora_mlp_cascade/log.txt
