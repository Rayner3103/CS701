#!/bin/bash
# Make F3_avg120-140_<split>_shift-0.2.zip: average the trainable weights of ckpt_e120/e130/e140 into one model
# (one network, one forward pass per image, no TTA), predict, move every box by -0.2 px, check the zip.
#   ./make_submission_F3_avg120-140.sh                    runs/F3_full150 from train_F3_full150.sh
#   RUN=/path/to/F3-S3-full150 ./make_submission_F3_avg120-140.sh
#   IMAGES=.../cs701-sar-course-data/test ./make_submission_F3_avg120-140.sh    Phase 2
#   EPOCHS="130" ./make_submission_F3_avg120-140.sh      a single checkpoint instead (F3 e130 = 52.38)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${RUN:-$HERE/runs/F3_full150}
IMAGES=${IMAGES:-$HERE/../../../cs701-sar-course-data/val}
EPOCHS=${EPOCHS:-120 130 140}
PY=${PY:-python}
AVG=$HERE/weights/avg_$(echo $EPOCHS | tr ' ' '-')
OUT=${OUT:-$HERE/F3_avg$(echo $EPOCHS | tr ' ' '-')_$(basename "$IMAGES")_shift-0.2.zip}
mkdir -p "$AVG"
cp "$RUN/metrics.json" "$RUN/log.txt" "$AVG/"  # predict.py rebuilds the model from metrics.json's args
$PY "$HERE/avg_checkpoints.py" "$AVG/model.pt" $(for e in $EPOCHS; do echo "$RUN/ckpt_e$e.pt"; done)
cd "$HERE/code"
$PY predict.py --run "$AVG" --images "$IMAGES" --shift -0.2 --out "$OUT"
$PY tools/check_submission.py "$OUT" "$IMAGES/images.json"
