#!/bin/bash
# Make the submission from the averaged weights with the 56.49 post-processing:
#   RPN 2000 proposals, no objectness re-weighting (obj_pow 0), soft-NMS sigma 0.5, mean of the 3 Cascade stages'
#   boxes, classification fused with the detector (lambda 2), boxes moved by -0.2 px, class-agnostic fusion
#   (box score = foreground probability x image-class probability, all 9 classes kept).
#   ./make_submission.sh                                   Phase 1 val images -> BP7_avgE100-180_val_agn.zip
#   IMAGES=.../cs701-sar-course-data/test ./make_submission.sh    Phase 2 test images
#   FROM_RAW=1 ./make_submission.sh                        no GPU: re-score weights/val_raw.pt (identical zip contents)
# Needs weights/model.pt and weights/metrics.json (model args). 1 GPU, about 1 min for 1,426 images.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGES=${IMAGES:-$HERE/../../../cs701-sar-course-data/val}
SPLIT=$(basename "$IMAGES")
OUT=${OUT:-$HERE/BP7_avgE100-180_${SPLIT}_agn.zip}
PY=${PY:-python}
POST="--rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --stage-box mean --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1 --topk 9"
cd "$HERE/code"
if [ -n "$FROM_RAW" ]; then
  $PY predict.py --run "$HERE/weights" --images "$IMAGES" --from-raw "$HERE/weights/${SPLIT}_raw.pt" $POST --out "$OUT"
else
  $PY predict.py --run "$HERE/weights" --weights model.pt --images "$IMAGES" --batch-size ${BATCH:-16} \
      --raw-out "$HERE/weights/${SPLIT}_raw_new.pt" $POST --out "$OUT"
fi
$PY tools/check_submission.py "$OUT" "$IMAGES/images.json"
