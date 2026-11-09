#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # all this account allows; shared by the 2 jobs (see W_BENCH / W_YOLO below)
#SBATCH --mem=32GB                  # 2 jobs at once
#SBATCH --gres=gpu:1                # One GPU (RTX 3090, 24 GB), shared by the 2 jobs
#SBATCH --time=00-24:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=END,FAIL        # When should you receive an email?
#SBATCH --output=seeds3_%j.out      # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                 # The partition you've been assigned
#SBATCH --account=cs701   # The account you've been assigned (normally student)
#SBATCH --qos=studentqos       # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=rayner.sim.2024@computing.smu.edu.sg # Who should receive the email notifications
#SBATCH --job-name=cs701-seeds3     # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# The missing seeds (target: 0, 1, 2) of every 24-epoch configuration: 69 runs, 36 configurations, in seeds3_jobs.tsv
# (about 121 h one after the other on a 3090). Two jobs run at once on the one GPU:
#   lane "bench": train.py runs (ViT / TerraMind / DINOv3 + Faster or Cascade R-CNN), GPU-bound
#   lane "yolo" : train_yolo.py runs (DINO-YOLO), data-loader (mosaic) bound
# The lanes pair a GPU-bound job with a CPU-bound one, so they compete least. Each lane takes its own kind of job first,
# longest first, then helps with the other lane's (jobs are claimed one at a time, so no job runs twice). A job is
# only started if it should still finish before the time limit; the rest wait for the next allocation. A job that is cut
# off resumes from its last epoch (--resume / weights/last.pt), and a finished job (state/done/<id>) is never run again.
# So: submit the chain of allocations with  bash gpuClusterRun/seeds3_submit.sh  and the runs get done in order.
#
# Submit from the repo (cs701-sar-benchmark):  sbatch gpuClusterRun/seeds3.sh
# Settings, as environment variables (sbatch exports them):
#   SMOKE=1        1-epoch smoke test of every configuration (one job each, seed = its first missing one, 256 images per
#                  split for train.py, 3% of train for train_yolo.py), into runs/seeds3_smoke. Use --time=03:00:00.
#                  The log ends with the status of every job and the peak GPU memory of the two lanes together.
#   LANES=2        jobs at once (1 = one after the other)
#   W_BENCH=1 W_YOLO=3   data-loader workers of a train.py / train_yolo.py job (defaults with 2 lanes; 1 lane: all CPUs)
#   ONLY=REGEX     only the jobs whose id matches, e.g. ONLY='^dinob_' or ONLY='-s1$'
#   WALL_H=23.5    hours after which no new job is started (a job needs est_h x PAD hours left)
#   PAD=1.7        expected slowdown of a job that shares the GPU (measured 1.5-1.6x for a light pair)
#   NO_GRAD_CKPT=safe|1|0  drop --dino-grad-ckpt of the ViT-B + YOLO runs: 18% faster per job (measured, 1.9 vs 1.6 it/s), same
#                  results, but about 3 GB more memory (at batch 16: 9.3-9.6 GB for LoRA r16/r64/r256 and unfreeze-2, 10.9 GB with
#                  MLP-LoRA, 12.0 GB with YOLO26-m; with it 6.2-8.2 GB). safe (default): only the five configs under 10 GB
#                  (NOCKPT_OK below); 1: all of them; 0: none. A pair of jobs then stays under 24 GB (at most about 22 GB).
#   EXTRA="..."    more arguments for every job (e.g. --batch-size 8 for a memory measurement)
#   TAG=name       SMOKE=1: a separate smoke folder runs/seeds3_smoke_<name>, e.g. to compare NO_GRAD_CKPT=1 with 0
#   DRY_RUN=1      only print the commands
#   OFFLINE=1      do not contact Hugging Face: the weights must already be in the cache (see below)
#   VENV=DIR       the virtual environment (default: the one of the other cluster scripts)
#   e.g.  SMOKE=1 sbatch --time=03:00:00 gpuClusterRun/seeds3.sh
#
# Once, on the login node (compute nodes may have no internet), with the venv active:
#   pip install -r requirements-yolo.txt                                    # also holds the train.py requirements
#   python -c "import timm; timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True)"   # DINOv3 ViT-B
#   python -c "import timm; timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True)"  # DINOv3 ViT-S
#   python -c "import timm; timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)"  # ViT (check train.py)
#   python -c "from ultralytics import YOLO; [YOLO(m) for m in ('yolo12s.pt','yolo26s.pt','yolo26m.pt')]"  # COCO weights
#   python -c "from terratorch.registry import BACKBONE_REGISTRY as R; R.build('terratorch_terramind_v1_base', pretrained=True, modalities=['S1GRD'])"
# (TerraMind weights, as sarbench/backbones.py builds it; if that call fails, run a 1-epoch `train.py --backbone terramind`
# on the login node instead.) The data, next to the repo
# (docs/setup.md): ../dataset/SARFact-Course-20K. The YOLO data and the oversampled data are built by this script.

set -uo pipefail   # no -e: one failed job must not stop the other jobs

command -v module >/dev/null 2>&1 && { module purge; module load Python/3.12.8-GCCcore-13.3.0; }
VENV="${VENV:-/common/home/users/r/rayner.sim.2024/repos/CS701/cs701-sar-benchmark/.venv}"
[ -d "$VENV/bin" ] && export PATH="$VENV/bin:$PATH"

# sbatch runs a copy of this script, so the repo is found from the directory the job was submitted from
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train.py ] || { [ -f ../train.py ] && cd ..; } || { echo "train.py not found: submit from cs701-sar-benchmark"; exit 1; }
REPO="$(pwd)"
JOBS="$REPO/gpuClusterRun/seeds3_jobs.tsv"
[ -f "$JOBS" ] || { echo "$JOBS not found"; exit 1; }

SMOKE="${SMOKE:-0}"
LANES="${LANES:-2}"
WALL_H="${WALL_H:-23.5}"
PAD="${PAD:-1.7}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-.}"
NOCKPT_OK='^(dinob_lora16|dinob_lora64|dinob_lora256|dinob_lora_unf2|os3_dinob_lora256)-'   # < 10 GB without checkpointing
CPUS="${SLURM_CPUS_PER_TASK:-4}"
# data-loader workers per job. One job alone gets all CPUs. Two jobs share them: train.py loads pre-resized images (light),
# train_yolo.py does mosaic augmentation on the CPU (heavy), so the yolo lane gets most of them.
if [ "$LANES" -ge 2 ]; then W_BENCH="${W_BENCH:-1}"; W_YOLO="${W_YOLO:-$(( CPUS > 1 ? CPUS - 1 : 1 ))}"
else W_BENCH="${W_BENCH:-$CPUS}"; W_YOLO="${W_YOLO:-$CPUS}"; fi
WORKERS=$W_BENCH
WALL_S=$(awk "BEGIN{printf \"%d\", $WALL_H * 3600}")
DATA="${DATA:-$REPO/../dataset}"
JOBID="${SLURM_JOB_ID:-local$$}"
if [ "$SMOKE" = 1 ]; then ROOT="runs/seeds3_smoke${TAG:+_$TAG}"; else ROOT="runs/seeds3"; fi
STATE="$ROOT/state"
CLAIM="$STATE/claims_$JOBID"      # one folder per job claimed in this allocation (mkdir is atomic: no job runs twice)
LOGS="$ROOT/logs"
mkdir -p "$STATE/done" "$CLAIM" "$LOGS" "$ROOT/yolo"

if [ "$DRY_RUN" != 1 ]; then
    python -c "import torch, timm, terratorch, ultralytics, pycocotools" 2>/dev/null \
        || { echo "missing Python packages in the venv: pip install -r requirements-yolo.txt"; exit 1; }
    for w in yolo12s.pt yolo26s.pt yolo26m.pt; do
        [ -f "$w" ] || { echo "$w not found in $REPO: see the login-node steps at the top of this script"; exit 1; }
    done
    [ -d "$DATA/SARFact-Course-20K/train" ] || { echo "no prepared data in $DATA/SARFact-Course-20K: see docs/setup.md"; exit 1; }
    [ "${OFFLINE:-0}" = 1 ] && export HF_HUB_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less memory fragmentation
    export OMP_NUM_THREADS=1   # the data-loader workers are the parallelism; more threads per process would oversubscribe the CPUs
fi

echo "job $JOBID on $(hostname), started $(date '+%F %T'), smoke=$SMOKE lanes=$LANES cpus=$CPUS workers: bench=$W_BENCH yolo=$W_YOLO"
echo "repo $REPO, runs in $ROOT, wall budget ${WALL_H} h, pad $PAD"
[ "$DRY_RUN" = 1 ] || nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
[ "$DRY_RUN" = 1 ] || python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to torch'"

# ---- data the jobs need, built once (symlinks and labels only, about a minute each) ----
build_data() {
    [ "$DRY_RUN" = 1 ] && return 0
    [ -d "$DATA/SARFact-YOLO/train" ] \
        || python prepare_yolo_data.py --src "$DATA/SARFact-Course-20K" --dst "$DATA/SARFact-YOLO" || return 1
    local arm opts
    for arm in os2:"--oversample 2 --paste-prob 0" os2paste:"--oversample 2 --paste-prob 0.5" \
               os2pasteall:"--oversample 2 --paste-prob 0.5 --paste-mode all" os3:"--oversample 3 --paste-prob 0"; do
        opts="${arm#*:}"; arm="${arm%%:*}"
        grep -q -P "\tyolo-$arm\t" "$JOBS" || continue
        if [ ! -f "$DATA/SARFact-YOLO-aug-$arm/data_aug.yaml" ]; then
            # shellcheck disable=SC2086
            python augment_small_objects.py $opts --dst "$DATA/SARFact-YOLO-aug-$arm" || return 1
        fi
    done
    if [ ! -f "$DATA/SARFact-Course-20K-aug-os3/detection/instances_train.json" ]; then
        python yolo_aug_to_course.py --yolo "$DATA/SARFact-YOLO-aug-os3" --dst "$DATA/SARFact-Course-20K-aug-os3" || return 1
    fi
}
build_data || { echo "building the data failed"; exit 1; }

yolo_data() {   # data key -> data.yaml
    case "$1" in
        yolo) echo "$DATA/SARFact-YOLO/data.yaml" ;;
        yolo-*) echo "$DATA/SARFact-YOLO-aug-${1#yolo-}/data_aug.yaml" ;;
    esac
}

# ---- one job ----
run_job() {   # id kind seed data name args
    local id=$1 kind=$2 seed=$3 data=$4 name=$5 args=$6 cmd out run run_name ep ev extra
    case "${NO_GRAD_CKPT:-safe}" in
        1) args="${args//--dino-grad-ckpt/}" ;;
        safe) [[ "$id" =~ $NOCKPT_OK ]] && args="${args//--dino-grad-ckpt/}" ;;
    esac
    if [ "$SMOKE" = 1 ]; then ep=1; ev=1; else ep=24; ev=4; fi
    if [ "$kind" = bench ]; then
        case "$data" in os3) out="$ROOT/os3/seed$seed"; data="$DATA/SARFact-Course-20K-aug-os3" ;;
                        *)   out="$ROOT/seed$seed";     data="$DATA/SARFact-Course-20K" ;; esac
        extra=""; [ "$SMOKE" = 1 ] && extra="--limit 256"
        # --resume: continues from last_checkpoint.pt, or starts fresh when there is none
        cmd="python train.py $args --epochs $ep --eval-every $ev --batch-size 8 --seed $seed --workers $W_BENCH --data $data --out $out --resume $extra ${EXTRA:-}"
    else
        run_name="${name}"; [ "$seed" != 0 ] && run_name="${name}_seed$seed"; [ "$SMOKE" = 1 ] && run_name="${run_name}_smoke"
        out="$ROOT/yolo"; run="$out/$run_name"
        extra=""; [ "$SMOKE" = 1 ] && extra="--fraction 0.03"
        [ -f "$run/weights/last.pt" ] && [ ! -f "$run/metrics.json" ] && extra="$extra --resume"
        cmd="python train_yolo.py $args --epochs $ep --seed $seed --workers $W_YOLO --data $(yolo_data "$data") --out $out --name $run_name $extra ${EXTRA:-}"
        [ -f "$run/metrics.json" ] && cmd="true # $run is finished (metrics.json)"
    fi
    local t0=$SECONDS e0 status=ok
    e0=$(date +%s)
    echo "$(date '+%T') START $id: $cmd"
    if [ "$DRY_RUN" = 1 ]; then return 0; fi
    # plain python, not srun: two srun steps would each wait for the one GPU
    # shellcheck disable=SC2086
    eval "$cmd" > "$LOGS/$id.log" 2>&1 && touch "$STATE/done/$id" || status=FAILED
    echo "$(date '+%T') $status $id after $((SECONDS - t0)) s (log $LOGS/$id.log)"
    echo -e "$id\t$status\t$((SECONDS - t0))\t$e0\t$(date +%s)" >> "$STATE/jobs_$JOBID.log"
}

# ---- one lane: its own kind of job first, then the other kind, longest first ----
worker() {   # preferred kind
    local pref=$1 other id kind est gb seed smoke data name args need left claimed
    [ "$pref" = bench ] && other=yolo || other=bench
    while :; do
        claimed=""
        for k in "$pref" "$other"; do
            while IFS=$'\t' read -r id kind est gb seed smoke data name args; do
                [ "$kind" = "$k" ] || continue
                [[ "$id" =~ $ONLY ]] || continue
                [ -e "$STATE/done/$id" ] && continue
                if [ "$SMOKE" = 1 ]; then
                    [ "$smoke" = 1 ] || continue
                else   # only start what should finish in time; shorter jobs further down the list may still fit
                    need=$(awk "BEGIN{printf \"%d\", $est * $PAD * 3600}"); left=$((WALL_S - SECONDS))
                    [ "$need" -le "$left" ] || continue
                fi
                mkdir "$CLAIM/$id" 2>/dev/null || continue
                run_job "$id" "$kind" "$seed" "$data" "$name" "$args"
                claimed=1; break 2
            done < <(grep -v '^#' "$JOBS")
        done
        [ -n "$claimed" ] || break
    done
}

# ---- peak GPU memory of the two lanes together ----
if [ "$DRY_RUN" != 1 ]; then
    # memory.used of the whole GPU (CUDA contexts and the allocator's cache included) every second: epoch seconds,MiB
    BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    ( while :; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"; sleep 1; done ) \
        > "$STATE/gpu_mem_$JOBID.csv" &
    SAMPLER=$!
fi

worker bench &
PIDS=$!
if [ "$LANES" -ge 2 ]; then worker yolo & PIDS="$PIDS $!"; fi
# shellcheck disable=SC2086
wait $PIDS
[ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null

echo "finished $(date '+%F %T') after $((SECONDS / 60)) min"
if [ -f "$STATE/jobs_$JOBID.log" ]; then
    echo "--- jobs of this allocation: id, status, seconds, peak GPU memory MiB while it ran (exact with LANES=1) ---"
    while IFS=$'\t' read -r id status secs e0 e1; do
        pk=$(awk -F, -v a="$e0" -v b="$e1" '$1 >= a && $1 <= b && $2 > m { m = $2 } END { print m + 0 }' "$STATE/gpu_mem_$JOBID.csv" 2>/dev/null)
        echo -e "$id\t$status\t$secs s\t$pk MiB"
    done < "$STATE/jobs_$JOBID.log"
    echo "GPU memory used before the jobs started: ${BASE:-?} MiB (other users/desktop; included in the numbers above)"
    echo "ok: $(grep -c -P '\tok\t' "$STATE/jobs_$JOBID.log"), failed: $(grep -c -P '\tFAILED\t' "$STATE/jobs_$JOBID.log")"
fi
[ -f "$STATE/gpu_mem_$JOBID.csv" ] && echo "peak GPU memory used (MiB), all jobs together: $(cut -d, -f2 "$STATE/gpu_mem_$JOBID.csv" | sort -n | tail -1)"
total=$(awk -F'\t' -v re="$ONLY" '!/^#/ && $1 ~ re && (smoke != 1 || $6 == 1)' smoke="$SMOKE" "$JOBS" | wc -l)
echo "jobs done so far: $(ls "$STATE/done" | wc -l) of $total (state in $STATE/done)"
