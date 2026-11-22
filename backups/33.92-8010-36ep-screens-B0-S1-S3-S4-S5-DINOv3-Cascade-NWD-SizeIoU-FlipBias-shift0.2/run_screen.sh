#!/bin/bash
# One 36-epoch screen on the team's local 80:10:10 split (SC1 recipe = SC2 at 36 epochs on local train), then the
# final post-processing on val/test (metrics_pp.json). Extra flags after NAME override the SC1 recipe (argparse: last wins).
#   bash team/run_screen.sh NAME [extra train.py flags]
set -e
cd "$(dirname "$0")"
NAME=$1; shift
DATA=${TEAM_DATA:-/content/team-data/SARFact-Course-20K}
OUT=${TEAM_OUT:-../runs/team}
python train.py --data "$DATA" --out "$OUT" --name "$NAME" \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 80 --lora-alpha 160 --cascade-rep 1536 --train-norms --backbone-lr 3e-4 --dump-raw --compile \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" \
  --epochs 36 --cascade-size-iou --flip-bias -0.3 --grad-ckpt --train-split train --eval-splits val,test "$@"
[ -n "$SKIP_PP" ] || python pp_eval.py --run "$OUT/$NAME"
