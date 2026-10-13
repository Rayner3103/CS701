#!/usr/bin/env bash
# The four pretrained adapter runs of run_all.sh (ViT and TerraMind, LoRA and MoE-LoRA), each run twice: without and
# with --spatial-prior. Batch size 8 fits a 10 GB RTX 3080.
# Every run uses the same seed (SEED, default 0), so the two runs of a pair differ only in the spatial prior.
# Everything goes to OUT_FOLDER (default runs): one folder per run, the combined log of this script
# (run_all_3080.log) and the settings of each launch (run_all_3080_details.txt).
# Usage: [SEED=N] [OUT_FOLDER=DIR] bash run_all_3080.sh [EPOCHS] [extra train.py arguments]
#   bash run_all_3080.sh                  # 24 epochs (the default of train.py)
#   bash run_all_3080.sh 12               # 12 epochs
#   OUT_FOLDER=runs/smoke bash run_all_3080.sh 1 --limit 64   # quick end-to-end check
#   SEED=1 OUT_FOLDER=runs/seed1 bash run_all_3080.sh 24      # all eight runs again with seed 1
# Resuming: launch again with the same OUT_FOLDER and EPOCHS. Each run's log.txt decides what happens to it:
#   - it ends with the final test results (after epoch EPOCHS, or earlier with --patience): complete, skipped;
#   - it stops in epoch n: the run resumes from last_checkpoint.pt, the last completed epoch (n - 1);
#   - no log, no checkpoint, or a log of a different number of epochs: the run starts over.
# Each configuration's pair runs back to back, so an interrupted script still leaves complete pairs to compare.
# A run that fails (e.g. CUDA out of memory) is reported and the next one still starts.
# Less memory fragmentation: TerraMind + MoE-LoRA (two backbone passes per batch) peaks at 8.7 of 9.6 GB with it,
# and runs out of memory without it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EPOCHS=${1:-24}
shift || true
SEED=${SEED:-0}
OUT_FOLDER=${OUT_FOLDER:-runs}
COMMON=(--batch-size 8 --epochs "$EPOCHS" --seed "$SEED" --out "$OUT_FOLDER")
EXTRA=("$@")
failed=()

# From here on, everything printed also goes to OUT_FOLDER/run_all_3080.log (appended: a resumed launch keeps the
# log of the earlier ones)
mkdir -p "$OUT_FOLDER"
exec > >(tee -a "$OUT_FOLDER/run_all_3080.log") 2>&1

{
    echo "started:     $(date '+%Y-%m-%d %H:%M:%S')"
    echo "command:     SEED=$SEED OUT_FOLDER=$OUT_FOLDER bash run_all_3080.sh $EPOCHS ${EXTRA[*]}"
    echo "epochs:      $EPOCHS"
    echo "seed:        $SEED"
    echo "extra args:  ${EXTRA[*]:-(none)}"
    echo "git:         $(git rev-parse --abbrev-ref HEAD 2>/dev/null) $(git rev-parse --short HEAD 2>/dev/null)" \
         "$( [ -n "$(git status --porcelain 2>/dev/null)" ] && echo '(uncommitted changes)')"
    echo "python:      $(command -v python)"
    echo "torch:       $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
    echo "gpu, driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1)"
} | tee -a "$OUT_FOLDER/run_all_3080_details.txt"

# What to do with one run, from its log.txt: prints "skip", "resume" or "fresh", a space, and the reason
run_status() {
    local dir=$1 log=$1/log.txt
    if [ ! -f "$log" ]; then
        echo "fresh no log.txt yet: starting"; return
    fi
    local logged_epochs last_epoch
    logged_epochs=$(head -1 "$log" | grep -oE '"epochs": [0-9]+' | grep -oE '[0-9]+$')
    last_epoch=$(grep -oE '^epoch [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')
    if [ "$logged_epochs" != "$EPOCHS" ]; then
        echo "fresh log.txt is from a run of ${logged_epochs:-?} epochs, not $EPOCHS: starting over"
    elif grep -q "^test (last, epoch " "$log"; then  # all EPOCHS epochs, or fewer with --patience
        echo "skip finished (test results in log.txt): skipped"
    elif [ -f "$dir/last_checkpoint.pt" ]; then
        echo "resume log.txt stops in epoch ${last_epoch:-1}: resuming from the last completed epoch"
    else
        echo "fresh log.txt stops in epoch ${last_epoch:-1} and there is no checkpoint: starting over"
    fi
}

run() {
    local backbone=$1 adapt=$2
    for prior in "" --spatial-prior; do
        local name=${backbone}_pretrained_${adapt}${prior:+_sp}
        local args=(--backbone "$backbone" --init pretrained --adapt "$adapt" $prior)
        local status reason
        read -r status reason <<< "$(run_status "$OUT_FOLDER/$name")"
        echo "=== $name: $reason"
        [ "$status" = skip ] && continue
        [ "$status" = resume ] && args+=(--resume)
        echo "=== train.py ${args[*]} ${COMMON[*]} ${EXTRA[*]}"
        python train.py "${args[@]}" "${COMMON[@]}" "${EXTRA[@]}" || failed+=("$name")
    done
}

run vit lora
run vit moelora
run terramind lora
run terramind moelora

echo "finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$OUT_FOLDER/run_all_3080_details.txt"
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed runs (launch again with the same OUT_FOLDER and EPOCHS to resume them):"
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
echo "All eight runs are complete."
