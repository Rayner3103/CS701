#!/usr/bin/env bash
# Small-object augmentation experiments (augment_small_objects.py), one GPU, one job after the other, 24 epochs each:
#   1. Build the four augmented YOLO datasets (dataset/SARFact-YOLO-aug-<arm>).
#   2. Ablation on the fine-tuned ViT-S + YOLO26-s (train_yolo.py defaults): the four arms. The baseline is the existing
#      dino_yolo26s_ep24 (+ _seed1), trained on the same split.
#   3. aug_pick_winner.py picks the arm with the best val delta_m. If none beats the baseline, the script stops here.
#   4. The winner on ViT-B/16 + LoRA r256 + YOLO26-s (runs/yolo/aug_<arm>_dinob_lora256_yolo26s_ep24).
#   5. The winner on DINOv3 + LoRA(attn+mlp) + Cascade R-CNN (runs/aug_cascade/<arm>), on a COCO-format copy of the arm
#      (yolo_aug_to_course.py). The unaugmented cascade run to compare with is runs/explore_seed0.
# Everything is scored on the original local val and test splits. Safe to launch again: a finished run is skipped, an
# unfinished one resumes from its last epoch. Log: runs/aug/run_aug.log.
# Usage: bash run_aug_experiments.sh [arm ...]      # default: all four arms
cd "$(dirname "$0")" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVS=/home/raynerubuntu/miniforge3/envs
PYY=$ENVS/cs701yolo/bin/python
PYB=$ENVS/cs701bench/bin/python
DATA=../dataset
ARMS=(os2paste os2 os3 os2pasteall)
[ $# -gt 0 ] && ARMS=("$@")
declare -A AUG=(
    [os2paste]="--oversample 2 --paste-prob 0.5"
    [os2]="--oversample 2 --paste-prob 0"
    [os3]="--oversample 3 --paste-prob 0"
    [os2pasteall]="--oversample 2 --paste-prob 0.5 --paste-mode all"
)
mkdir -p runs/aug
exec > >(tee -a runs/aug/run_aug.log) 2>&1
say() { echo "=== $(date '+%F %T') $*"; }

yolo_run() {  # name, then train_yolo.py arguments; scores on val and test at the end
    local name=$1; shift
    local run=runs/yolo/$name
    if [ -f "$run/metrics.json" ]; then say "skip $name (done)"; return 0; fi
    if [ -f "$run/results.csv" ] && [ "$(($(wc -l < "$run/results.csv") - 1))" -ge 24 ]; then
        say "$name trained but not scored: scoring"
        $PYY eval_yolo.py --run "$run" || echo "FAILED scoring $name"; return 0
    fi
    if [ -f "$run/weights/last.pt" ]; then
        say "resume $name"
        $PYY train_yolo.py "$@" --epochs 24 --name "$name" --resume || echo "FAILED $name"
    else
        say "start $name: $*"
        $PYY train_yolo.py "$@" --epochs 24 --name "$name" || echo "FAILED $name"
    fi
}

say "build datasets: ${ARMS[*]}"
for arm in "${ARMS[@]}"; do
    if [ ! -f "$DATA/SARFact-YOLO-aug-$arm/data_aug.yaml" ]; then
        # shellcheck disable=SC2086
        $PYY augment_small_objects.py ${AUG[$arm]} --dst "$DATA/SARFact-YOLO-aug-$arm" || { echo "FAILED building $arm"; exit 1; }
    fi
done

for arm in "${ARMS[@]}"; do
    yolo_run "aug_${arm}_dino_yolo26s_ep24" --yolo yolo26s --data "$DATA/SARFact-YOLO-aug-$arm/data_aug.yaml"
done

say "pick the winner"
$PYY aug_pick_winner.py "${ARMS[@]}" || { say "no augmented arm beats the baseline: stopping before the ViT-B and cascade runs"; exit 1; }
W=$(cat runs/aug/winner.txt)

yolo_run "aug_${W}_dinob_lora256_yolo26s_ep24" --yolo yolo26s --dino vit_base_patch16_dinov3.lvd1689m --dino-adapt lora \
    --lora-rank 256 --dino-grad-ckpt --data "$DATA/SARFact-YOLO-aug-$W/data_aug.yaml"

say "cascade on $W"
if [ ! -f "$DATA/SARFact-Course-20K-aug-$W/detection/instances_train.json" ]; then
    $PYB yolo_aug_to_course.py --yolo "$DATA/SARFact-YOLO-aug-$W" --dst "$DATA/SARFact-Course-20K-aug-$W" || { echo "FAILED conversion"; exit 1; }
fi
$PYB train.py --backbone dinov3 --init pretrained --adapt lora --lora-targets attn+mlp --cascade --cascade-checkpoint \
    --batch-size 8 --epochs 24 --seed 0 --resume --data "$DATA/SARFact-Course-20K-aug-$W" --out "runs/aug_cascade/$W" \
    || echo "FAILED cascade"
say "all done"
