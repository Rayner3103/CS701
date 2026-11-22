#!/bin/bash
# Checks run_full.sh end to end on 32 images: 2 epochs, then --resume to 4 epochs (as after a dropped runtime),
# each followed by the val submission and its format check.
set -e
cd "$(dirname "$0")"
export TEAM_OUT=../runs/team-smoke
rm -rf "$TEAM_OUT/full"
S3="--lora-rank 48 --lora-alpha 96 --cascade-rep 1024 --unfreeze-last 2 --limit 32 --last-every 1 --save-every 0"
bash run_full.sh full $S3 --epochs 2
bash run_full.sh full $S3 --epochs 4 --resume
grep "resumed from last.pt after epoch 1" "$TEAM_OUT/full/log.txt" && echo "RESUME OK"
ls -la "$TEAM_OUT/full"/*.zip
