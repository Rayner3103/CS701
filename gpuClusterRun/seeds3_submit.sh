#!/bin/bash
# Submit the seeds3 runs (gpuClusterRun/seeds3.sh) from the repo (cs701-sar-benchmark):
#   bash gpuClusterRun/seeds3_submit.sh smoke     # 1-epoch check of every configuration, ~1-2 h; look at its log first
#   bash gpuClusterRun/seeds3_submit.sh           # the real runs: a chain of 6 allocations of 24 h, one after the other
#   bash gpuClusterRun/seeds3_submit.sh 4         # a chain of 4
# Each allocation runs 2 jobs at once on the GPU and starts only what can finish before its time limit; the next one
# (--dependency=afterany) carries on with what is left, and resumes a job that was cut off. About 100-120 h of work
# in all, so 5-6 allocations; allocations with nothing left to do end after a few seconds.
# Extra settings (LANES, NO_GRAD_CKPT, ONLY, ...) are environment variables: see the top of seeds3.sh.
cd "$(dirname "$0")/.." || exit 1
if [ "${1:-}" = smoke ]; then
    SMOKE=1 sbatch --time=03:00:00 --job-name=cs701-seeds3-smoke gpuClusterRun/seeds3.sh
    exit
fi
N="${1:-6}"
prev=""
for i in $(seq 1 "$N"); do
    dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
    prev=$(sbatch --parsable $dep --job-name="cs701-seeds3-$i" gpuClusterRun/seeds3.sh) || exit 1
    echo "allocation $i: job $prev ${dep:+(after $(echo "$dep" | cut -d: -f2))}"
done
