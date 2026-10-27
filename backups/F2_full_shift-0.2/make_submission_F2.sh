#!/bin/bash
# Make F2_full_val_shift-0.2.zip: predict with the F2 weights, then move every box by -0.2 px.
#   ./make_submission_F2.sh                 weights in weights/ (the backup copy), else runs/F2_full (retrained)
#   IMAGES=.../cs701-sar-course-data/test ./make_submission_F2.sh    Phase 2
#   FROM_RAW=1 ./make_submission_F2.sh      no GPU: re-score weights/F2_full_val_raw.pt (identical zip contents)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGES=${IMAGES:-$HERE/../../../cs701-sar-course-data/val}
RUN=${RUN:-$HERE/weights}; [ -f "$RUN/model.pt" ] || RUN=$HERE/runs/F2_full
OUT=${OUT:-$HERE/F2_full_$(basename "$IMAGES")_shift-0.2.zip}
PY=${PY:-python}
cd "$HERE/code"
if [ -n "$FROM_RAW" ]; then
  $PY predict.py --run "$RUN" --images "$IMAGES" --from-raw "$HERE/weights/F2_full_val_raw.pt" --shift -0.2 --out "$OUT"
else
  $PY predict.py --run "$RUN" --images "$IMAGES" --shift -0.2 --out "$OUT"
fi
$PY tools/check_submission.py "$OUT" "$IMAGES/images.json"
