#!/bin/bash
# Pre-run for gpuClusterRun/junkai_best_os3.sh: run it ONCE on the login node (it needs internet, not a GPU), from anywhere:
#   bash gpuClusterRun/junkai_best_os3_prepare.sh
# Then submit yourself: sbatch --time=01:00:00 gpuClusterRun/junkai_best_os3.sh --smoke  (smoke first), then sbatch gpuClusterRun/junkai_best_os3.sh. Safe to run again: every step is skipped when done.
# It does:
#   1. python packages in the venv (junkai_os3/requirements.txt), only if something is missing
#   2. the data: unpack ../cs701-sar-course-data/train.tar.gz and convert it into ../dataset/SARFact-Course-20K
#      (the HuggingFace download must be there; images are symlinked, so keep cs701-sar-course-data in place)
#   3. the trainvl (8,400) / vallike (992) split files, built into that dataset folder
#   4. the DINOv3 ViT-B/16 weights into the Hugging Face cache (the job can then run with OFFLINE=1)
#   5. a check that the training loader sees the data (3x oversampled epoch size)
# Layout expected: CS701/{cs701-sar-benchmark, cs701-sar-course-data, dataset}   (docs/setup.md)
set -euo pipefail

module purge
module load Python/3.12.8-GCCcore-13.3.0
source "/common/home/users/r/rayner.sim.2024/repos/CS701/cs701-sar-benchmark/.venv/bin/activate"

cd "$(dirname "$0")/.."
[ -f junkai_os3/train.py ] || { echo "junkai_os3/train.py not found: is this the cs701-sar-benchmark repo on the rayner branch (git pull)?"; exit 1; }
RAW="${RAW:-$(pwd)/../cs701-sar-course-data}"
DATA="${DATA:-$(pwd)/../dataset/SARFact-Course-20K}"

echo "== 1. python packages"
if python -c "import torch, timm, terratorch, pycocotools" 2>/dev/null; then
    echo "all there"
else
    pip install -r junkai_os3/requirements.txt
    python -c "import torch, timm, terratorch, pycocotools"
fi

echo "== 2. data: $DATA"
if [ -f "$DATA/detection/instances_train.json" ] && [ -f "$DATA/classification/train.csv" ]; then
    echo "already converted"
else
    [ -d "$RAW" ] || { echo "no $RAW: hf download doem1997/cs701-sar-course-data --repo-type dataset --local-dir $RAW"; exit 1; }
    if [ ! -f "$RAW/train/labels.csv" ]; then
        [ -f "$RAW/train.tar.gz" ] || { echo "neither $RAW/train/ nor $RAW/train.tar.gz"; exit 1; }
        (cd "$RAW" && tar -xzf train.tar.gz)
    fi
    python prepare_course_data.py --src "$RAW" --dst "$DATA"   # 7,512 / 940 / 940, seed 0
fi

echo "== 3. trainvl / vallike split"
if [ -f "$DATA/classification/vallike.csv" ] && [ -f "$DATA/classification/trainvl.csv" ] \
        && [ -f "$DATA/detection/instances_vallike.json" ] && [ -f "$DATA/detection/instances_trainvl.json" ]; then
    echo "already built"
else
    python -I junkai_os3/tools/make_vallike_split.py --data "$DATA"
fi

echo "== 4. DINOv3 ViT-B/16 weights"
python -c "import timm; timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True); print('in the cache')"

echo "== 5. check"
(cd junkai_os3 && python -I - "$DATA" <<'PY'
import sys
sys.path.insert(0, '.')
from sarbench.data import SARMultiTask
data = sys.argv[1]
plain, over = (SARMultiTask(data, 'trainvl', train=True, oversample_small=n) for n in (1, 3))
vallike = SARMultiTask(data, 'vallike')
print(f'trainvl {len(plain)} images ({-(-len(plain) // 16)} steps/epoch at batch 16), with os3 {len(over)} items '
      f'({-(-len(over) // 16)} steps/epoch), vallike {len(vallike)}')
assert (len(plain), len(vallike)) == (8400, 992), 'unexpected split sizes'
PY
)
echo "ready: sbatch --time=01:00:00 gpuClusterRun/junkai_best_os3.sh --smoke"
