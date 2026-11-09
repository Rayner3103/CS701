#!/bin/bash
# Submit Junkai's best model + 3x small-object oversampling (gpuClusterRun/junkai_best_os3.sh) from the repo (cs701-sar-benchmark):
#   bash gpuClusterRun/junkai_best_os3_submit.sh smoke    # 3 epochs on 64 images, then average / predict / score: look at its log first
#   bash gpuClusterRun/junkai_best_os3_submit.sh          # the real run (OS=3): a chain of 4 allocations of 24 h, one after the other
#   bash gpuClusterRun/junkai_best_os3_submit.sh 3        # a chain of 3
#   OS=1 bash gpuClusterRun/junkai_best_os3_submit.sh     # the control without oversampling (needs about half the time: a chain of 2)
# Each allocation trains for up to 23.5 h and stops; the next one (--dependency=afterany) resumes from the last epoch
# (training_state.pt is saved every epoch). The allocation that reaches epoch 180 also averages e100-e180, predicts vallike and
# writes delta_m_vallike.json; allocations with nothing left to do end after a few seconds. The oversampled epochs are about
# 2.0x longer than plain ones (about 40+ GPU-hours in all, so 2-3 allocations, 4 leaves room for a slower GPU).
# Settings (OS, SEED, DATA, OFFLINE, WALL_H, EXTRA, ...) are environment variables: see the top of junkai_best_os3.sh.
cd "$(dirname "$0")/.." || exit 1
if [ "${1:-}" = smoke ]; then
    SMOKE=1 sbatch --time=01:00:00 --job-name=junkai-os3-smoke gpuClusterRun/junkai_best_os3.sh
    exit
fi
N="${1:-4}"
prev=""
for i in $(seq 1 "$N"); do
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    prev=$(sbatch --parsable $dep --job-name="junkai-os${OS:-3}-$i" gpuClusterRun/junkai_best_os3.sh) || exit 1
    echo "allocation $i: job $prev ${dep:+(after $(echo "$dep" | cut -d: -f2))}"
done
