#!/bin/bash
# Score SC1 on the local val / test holdouts under the final post-processing, from the saved raw outputs (CPU, minutes).
#   ./eval_SC1.sh                 weights/ (the backup copy), else runs/SC1_r80h1536 (retrained)
# Writes RUN/metrics_pp.json; compare it with run/metrics_pp.json.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=${RUN:-$HERE/weights}; [ -f "$RUN/raw_val.pt" ] || RUN=$HERE/runs/SC1_r80h1536
PY=${PY:-python}
cd "$HERE/code"
$PY pp_eval.py --run "$RUN"
