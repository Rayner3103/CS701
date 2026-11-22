#!/bin/bash
# Make SC2_e130_<split>_shift-0.2.zip: predict with the epoch-130 weights, every box moved by -0.2 px.
#   ./make_submission_SC2_e130.sh               weights/model.pt (= ckpt_e130.pt), else runs/SC2_full150/ckpt_e130.pt
#   IMAGES=.../cs701-sar-course-data/test ./make_submission_SC2_e130.sh    Phase 2
#   FROM_RAW=1 ./make_submission_SC2_e130.sh    no GPU: re-score weights/SC2_e130_val_raw.pt (identical zip contents)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGES=${IMAGES:-$HERE/../../../cs701-sar-course-data/val}
if [ -f "$HERE/weights/model.pt" ]; then RUN=$HERE/weights; W=model.pt; else RUN=$HERE/runs/SC2_full150; W=ckpt_e130.pt; fi
OUT=${OUT:-$HERE/SC2_e130_$(basename "$IMAGES")_shift-0.2.zip}
PY=${PY:-python}
cd "$HERE/code"
if [ -n "$FROM_RAW" ]; then
  $PY predict.py --run "$RUN" --images "$IMAGES" --from-raw "$HERE/weights/SC2_e130_val_raw.pt" --shift -0.2 --out "$OUT"
else
  $PY predict.py --run "$RUN" --weights $W --images "$IMAGES" --shift -0.2 --out "$OUT"
fi
$PY tools/check_submission.py "$OUT" "$IMAGES/images.json"
