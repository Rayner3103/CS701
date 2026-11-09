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
#SBATCH --time=01-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=END,FAIL        # When should you receive an email?
#SBATCH --output=junkai_best_os3_%j.out # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=cs701   # The account you've been assigned (normally student)
#SBATCH --qos=studentqos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=rayner.sim.2024@computing.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=junkai-best-os3       # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Junkai's best model (origin/junkai, backups/final/materials: Phase 1 val delta_m 56.49, Phase 2 test 61.25) retrained from
# scratch with the rayner branch's 3x small-object oversampling ("os3": every training image with a box under 32^2 px at
# 512 appears 3 times, originals kept; junkai_os3/ is a copy of that model's code plus the new --oversample-small flag).
# It trains on `trainvl` (8,400 images) and is scored on `vallike` (992 images), the split of Junkai's report, so the
# delta_m is comparable with his Table 2 / Table 3 (official val references 93.31 / 26.13: comparable between runs on this
# split, not with the leaderboard). OS=1 is the control: the same recipe without oversampling.
#
# Everything that improved delta_m is in:
#   model     DINOv3 ViT-B/16 frozen + LoRA r80 / alpha 160 on attn.qkv, attn.proj, mlp.fc1, mlp.fc2 (LayerNorms train),
#             ViTDet pyramid + 64-channel spatial prior, anchors 6-181 px matched by max(IoU, NWD C=8), anchor offset 0.5,
#             3-stage Cascade R-CNN width 1536 with size-aware IoU thresholds (a4), aligned RoIAlign   (39.87 M trainable)
#   training  flips with --flip-bias -0.3; AdamW head lr 1e-4, backbone lr 3e-4, wd 0.05; 1 warm-up epoch, then a 200-epoch
#             cosine, stopped after epoch 180 with a checkpoint every 10; global batch 16; gradient checkpointing
#   after     average of the checkpoints e100, 110, ..., 180 (one model, one forward pass); RPN top 2000, objectness^0,
#             soft-NMS sigma 0.5, mean of the 3 stage boxes, shift -0.2 px, class-agnostic fusion (gamma 1, topk 9),
#             classification lambda 2, min_keep 100, lattice snapping (integer corners for aircraft / car / 256 px cells,
#             per-image lattice for wind turbines)
# One epoch walks the oversampled list (about 2.0x the images of the plain epoch), so this is about 2x the compute of
# Junkai's recipe: ~40+ GPU-hours on one GPU, i.e. a chain of 2-3 allocations of 24 h (a job cut off resumes from its
# last epoch: training_state.pt is saved every epoch). Rough per-epoch time on one RTX 6000 Ada: 6-7 min plain, 13 min os3.
#
# Submit from the repo (cs701-sar-benchmark), after the pre-run (junkai_best_os3_prepare.sh):
#   sbatch --time=01:00:00 gpuClusterRun/junkai_best_os3.sh --smoke   # 3 epochs on 64 images, then average / predict / score: ~30 min
#   sbatch gpuClusterRun/junkai_best_os3.sh                           # the real run, OS=3: trains up to 23.5 h, then stops
#   OS=1 sbatch gpuClusterRun/junkai_best_os3.sh                      # the control without oversampling (about half the time)
# One allocation trains for at most WALL_H hours and stops; submit it again (or chain it) until the log says "finished" and
# delta_m_vallike.json exists: every allocation resumes from the last epoch and one that has nothing left ends in seconds.
# Chain, for example 3 allocations one after the other:
#   j=$(sbatch --parsable gpuClusterRun/junkai_best_os3.sh)
#   j=$(sbatch --parsable --dependency=afterany:$j gpuClusterRun/junkai_best_os3.sh)
#   j=$(sbatch --parsable --dependency=afterany:$j gpuClusterRun/junkai_best_os3.sh)
# Settings, as environment variables (sbatch exports them); do not change them between the allocations of one run:
#   OS=3           oversampling factor of the small-object images (1: control)
#   SEED=0         training seed
#   --smoke        (argument after the script name, or SMOKE=1) 3-epoch pipeline check on 64 images, into runs/junkai_os3_smoke
#   DATA=DIR       SARFact-Course-20K (default ../dataset/SARFact-Course-20K), prepared by the pre-run
#   OFFLINE=1      do not contact Hugging Face: the DINOv3 weights must already be in the cache (see below)
#   WALL_H=23.5    hours of training per allocation; the rest of the 24 h is for averaging, prediction and scoring
#   EXTRA="..."    more arguments for train.py (e.g. EXTRA="--batch-size 8" if the GPU is too small for batch 16 with gradient
#                  checkpointing: this changes the recipe)
#
# Pre-run, once, on the login node (internet, no GPU): bash gpuClusterRun/junkai_best_os3_prepare.sh
#   (venv packages, data conversion into ../dataset/SARFact-Course-20K, trainvl / vallike split, DINOv3 weights, a check).
# This script then only trains and scores. If it fails with a connection error, resubmit with OFFLINE=1.
# Results, in runs/junkai_os3/<name>/ :  delta_m_vallike.json (F1, mAP, AP50, delta_m with and without snapping, per-class AP),
# avg_model.pt (the averaged weights), raw_vallike.pt, predict_vallike.log (pycocotools cross-check), log.txt (training).

set -euo pipefail
T0=$(date +%s)
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        *) echo "unknown argument $arg (only --smoke)"; exit 2 ;;
    esac
done

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Python/3.12.8-GCCcore-13.3.0

# This command assumes that you've already created the environment previously
# (created and filled by the pre-run, or: python3.12 -m venv .venv && pip install -r junkai_os3/requirements.txt)
source "/common/home/users/r/rayner.sim.2024/repos/CS701/cs701-sar-benchmark/.venv/bin/activate"

# sbatch runs a copy of this script, so the repo is found from the directory the job was submitted from
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f junkai_os3/train.py ] || { [ -f ../junkai_os3/train.py ] && cd ..; } || { echo "junkai_os3/train.py not found: submit from cs701-sar-benchmark"; exit 1; }
REPO=$(pwd)
python -c "import torch, timm, terratorch, pycocotools" 2>/dev/null \
    || { echo "missing Python packages in the venv: run gpuClusterRun/junkai_best_os3_prepare.sh on the login node"; exit 1; }
DATA="${DATA:-$REPO/../dataset/SARFact-Course-20K}"
[ -d "$DATA/detection" ] || { echo "no $DATA: run gpuClusterRun/junkai_best_os3_prepare.sh on the login node first"; exit 1; }

OS="${OS:-3}"
SEED="${SEED:-0}"
WALL_H="${WALL_H:-23.5}"
[ "${OFFLINE:-0}" = "1" ] && export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less memory fragmentation
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# The recipe's schedule: 200-epoch cosine, stop after epoch 180, average e100-e180 (checkpoint every 10)
EPOCHS=200; STOP=180; SAVE_EVERY=10; EVAL_EVERY=20; AVG="$(seq 100 10 180 | tr '\n' ' ')"
OUT=runs/junkai_os3; LIMIT=""
if [ "${SMOKE:-0}" = "1" ]; then
    EPOCHS=3; STOP=3; SAVE_EVERY=1; EVAL_EVERY=1; AVG="1 2"; OUT=runs/junkai_os3_smoke; LIMIT="--limit 64"
fi
NAME="junkai_best_os${OS}_trainvl_seed${SEED}"
RUN="$OUT/$NAME"
WORKERS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$OUT"

echo "job ${SLURM_JOB_ID:-local} on $(hostname), started $(date '+%F %T')"
echo "repo $REPO, data $DATA, run $RUN, oversample-small x$OS, seed $SEED"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'"

if [ -f "$RUN/delta_m_vallike.json" ]; then
    echo "$RUN is finished (delta_m_vallike.json): nothing to do"
    exit 0
fi

# the data and the trainvl / vallike split come from the pre-run (login node): gpuClusterRun/junkai_best_os3_prepare.sh
for f in classification/vallike.csv classification/trainvl.csv detection/instances_vallike.json detection/instances_trainvl.json; do
    [ -f "$DATA/$f" ] || { echo "missing $DATA/$f: run  bash gpuClusterRun/junkai_best_os3_prepare.sh  on the login node first"; exit 1; }
done

cd junkai_os3   # its own sarbench/ must be the one imported, not the repo root's
RUN="../$RUN"; OUT="../$OUT"

# Training, until epoch $STOP or until WALL_H hours of this allocation are used (then the next allocation resumes)
done_training() { [ -f "$RUN/metrics.json" ] || { [ -f "$RUN/log.txt" ] && grep -q "Stopped cleanly after epoch $STOP" "$RUN/log.txt"; }; }
if done_training; then
    echo "training of $NAME is complete"
else
    BUDGET=$(python -c "print(max(60, int($WALL_H * 3600 - ($(date +%s) - $T0))))")
    RESUME=""
    if [ -f "$RUN/training_state.pt" ]; then RESUME="--resume $RUN/training_state.pt"; echo "resuming from $RUN/training_state.pt"; fi
    echo "training for at most $BUDGET s"
    set +e
    # shellcheck disable=SC2086  # EXTRA and the option groups are meant to split into arguments
    timeout -s TERM -k 300 "$BUDGET" srun --gres=gpu:1 python train.py \
        --data "$DATA" --out "$OUT" --name "$NAME" --train-split trainvl --eval-splits vallike --seed "$SEED" \
        --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
        --lora-rank 80 --lora-alpha 160 --train-norms --lr 1e-4 --backbone-lr 3e-4 --weight-decay 0.05 \
        --spatial-prior 64 --cascade --cascade-rep 1536 --cascade-size-iou --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 \
        "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --flip-bias -0.3 --grad-ckpt \
        --warmup-epochs 1 --epochs "$EPOCHS" --schedule-epochs "$EPOCHS" --stop-after-epoch "$STOP" \
        --save-every "$SAVE_EVERY" --save-training-state --reproducible-loader \
        --batch-size 16 --workers "$WORKERS" --eval-every "$EVAL_EVERY" --eval-batch-size 4 \
        --oversample-small "$OS" $LIMIT ${EXTRA:-} $RESUME
    rc=$?
    set -e
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        echo "time budget used up ($(date '+%F %T')): the next allocation of the chain resumes from the last epoch"
        exit 0
    elif [ $rc -ne 0 ]; then
        echo "train.py failed (exit $rc)"; exit $rc
    fi
    done_training || { echo "train.py ended without reaching epoch $STOP"; exit 1; }
fi

# Average the checkpoints, predict vallike with the FINAL post-processing, score (F1, mAP, delta_m)
if [ ! -f "$RUN/avg_model.pt" ]; then
    # shellcheck disable=SC2086
    python tools/average_local.py --dir "$RUN" --epochs $AVG --out "$RUN/avg_model.pt"
fi
if [ ! -f "$RUN/raw_vallike_avg.pt" ]; then
    # --split vallike prints the pycocotools score of the post-processed boxes without snapping: a cross-check of local_eval.py
    python predict.py --run "$RUN" --weights avg_model.pt --data "$DATA" --split vallike --dump-raw \
        --batch-size 4 --workers 2 --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --stage-box mean --cls-lambda 2 \
        --shift -0.2 --fusion agn --gamma 1 --topk 9 --min-keep 100 2>&1 | tee "$RUN/predict_vallike.log"
    mv "$RUN/raw_vallike.pt" "$RUN/raw_vallike_avg.pt"
fi
python tools/local_eval.py --raw "$RUN/raw_vallike_avg.pt" --data "$DATA" --split vallike --out "$RUN/delta_m_vallike.json"
echo "finished $(date '+%F %T'); results in $RUN (delta_m_vallike.json, avg_model.pt)"

# next to the other arm of the comparison (same seed), if that run has finished too
python - "$OUT" "$SEED" <<'PY'
import json, sys
from pathlib import Path
out, seed = Path(sys.argv[1]), sys.argv[2]
r = {}
for os_ in (1, 3):
    f = out / f'junkai_best_os{os_}_trainvl_seed{seed}' / 'delta_m_vallike.json'
    if f.exists():
        r[os_] = json.loads(f.read_text())['FINAL (with snapping)']
        print(f"os{os_}: delta_m {r[os_]['delta_m']:+.2f}  macro-F1 {r[os_]['macro_f1']:.2f}  mAP {r[os_]['mAP']:.2f}")
if len(r) == 2:
    print(f"os3 - os1: delta_m {r[3]['delta_m'] - r[1]['delta_m']:+.2f}  mAP {r[3]['mAP'] - r[1]['mAP']:+.2f} "
          '(seed noise of one run: about +-0.7 delta_m on the short C36 runs)')
PY
