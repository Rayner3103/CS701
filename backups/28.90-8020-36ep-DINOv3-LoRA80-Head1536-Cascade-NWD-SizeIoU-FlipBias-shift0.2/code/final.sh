#!/bin/bash
# Final model: train the chosen recipe on ALL labelled images (local train + val + test holdouts = the 9,392 official
# train images; never the Codabench val/test images), then predict the Codabench images and write submission.zip.
# usage: bash final.sh NAME SPLIT_DIR "TRAIN_ARGS"   e.g.
#   bash final.sh FINAL_A ~/cs701_data/cs701-sar-course-data/val "--backbone dinov3 --init pretrained --adapt lora ..."
set -e
NAME=$1; IMAGES=$2; ARGS=$3
PY=~/miniconda3/envs/cs701bench/bin/python
cd "$(dirname "$0")"
$PY train.py --data $HOME/cs701_data/dataset/SARFact-Course-20K --out $HOME/cs701_runs --name $NAME \
    --train-split train+val+test --eval-splits '' --save full $ARGS
mkdir -p submissions
$PY predict.py --run $HOME/cs701_runs/$NAME --images $IMAGES --out submissions/$NAME.zip
$PY - "$IMAGES" "submissions/$NAME.zip" << 'PY'
import csv, io, json, math, sys, zipfile
images = json.load(open(sys.argv[1] + '/images.json'))['images']
z = zipfile.ZipFile(sys.argv[2])
rows = list(csv.DictReader(io.StringIO(z.read('classification.csv').decode())))
dets = json.loads(z.read('detection.json'))
ids = {im['id'] for im in images}
assert len(rows) == len(images) and {int(r['image_id']) for r in rows} == ids, 'classification rows'
assert all(0 <= int(r['label']) <= 8 for r in rows), 'labels'
assert all(d['image_id'] in ids and 1 <= d['category_id'] <= 9 and d['bbox'][2] > 0 and d['bbox'][3] > 0
           and all(math.isfinite(v) for v in d['bbox'] + [d['score']]) for d in dets), 'detections'
print(f'RESULT final submission format OK: {len(rows)} rows, {len(dets)} detections -> {sys.argv[2]}')
PY
