#!/bin/bash
# Full-data run (all 9,392 course train images, no holdout, fixed last epoch) with the SC2 recipe, then the Codabench
# val submissions of epochs 130/140/150 (shift -0.2 as SC2). Extra flags after NAME override the recipe; the worker
# appends --resume after a dropped runtime (train.py continues from last.pt, saved every 2 epochs).
#   bash team/run_full.sh NAME [extra train.py flags]
set -e
cd "$(dirname "$0")"
NAME=$1; shift
DATA=${TEAM_DATA:-/content/team-data/SARFact-Course-20K}
OUT=${TEAM_OUT:-../runs/team}
VAL=${TEAM_VAL:-../data/val}
python train.py --data "$DATA" --out "$OUT" --name "$NAME" \
  --backbone dinov3 --init pretrained --adapt lora --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 \
  --lora-rank 80 --lora-alpha 160 --cascade-rep 1536 --train-norms --backbone-lr 3e-4 --compile \
  --cascade --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 "--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" \
  --cascade-size-iou --flip-bias -0.3 --grad-ckpt --train-split train+val+test --eval-splits '' --save-every 10 \
  --epochs 150 "$@"
RUN="$OUT/$NAME"
python predict.py --run "$RUN" --images "$VAL" --shift -0.2 --out "$RUN/submission-val-e150-shift-0.2.zip"
python tools/check_submission.py "$RUN/submission-val-e150-shift-0.2.zip" "$VAL/images.json"
for e in 140 130; do
  [ -f "$RUN/ckpt_e$e.pt" ] || continue
  python predict.py --run "$RUN" --weights "ckpt_e$e.pt" --images "$VAL" --shift -0.2 --out "$RUN/submission-val-e$e-shift-0.2.zip"
  python tools/check_submission.py "$RUN/submission-val-e$e-shift-0.2.zip" "$VAL/images.json"
done
echo "RESULT $NAME done"
