#!/bin/bash
# Builds and trains every screen config for 1 epoch on 32 images: train.py refuses an over-budget model, so this is
# the real parameter check (each run prints its "parameters:" line). Keep the flags in sync with colab/team-queue.json.
set -e
cd "$(dirname "$0")"
export TEAM_OUT=../runs/team-smoke SKIP_PP=1
mkdir -p "$TEAM_OUT" && : > "$TEAM_OUT/params.txt"
for cfg in "B0" "S1 --spatial-prior 64" "S2 --lr 2e-4 --backbone-lr 6e-4" \
           "S3 --lora-rank 48 --lora-alpha 96 --cascade-rep 1024 --unfreeze-last 2" \
           "S4 --spatial-prior 64 --pretrained-prior --lora-rank 77 --lora-alpha 154"; do
  set -- $cfg
  bash run_screen.sh "$@" --epochs 1 --limit 32
  echo "$cfg: $(grep -E "^parameters" "$TEAM_OUT/$1/log.txt")" | tee -a "$TEAM_OUT/params.txt"
done
python pp_eval.py --run "$TEAM_OUT/B0" || echo "pp_eval failed on the 32-image dump (check imports/paths only)"
echo "SMOKE OK"
