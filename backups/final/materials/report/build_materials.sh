#!/bin/bash
# Rebuilds backups/final/materials: text records and code are pushed; weights, raw outputs, predictions, data and session
# transcripts stay in materials/local_only (hard links, ignored by git, no extra disk). Safe to rerun.
set -eu
ROOT=/mnt/hdd/junkaichen/CS701/CS701; RUNS=/mnt/hdd/junkaichen/CS701/cs701_runs; M=$ROOT/backups/final/materials
SP=/tmp/claude-1001/-mnt-hdd-junkaichen-CS701-CS701/bb21cad1-800d-4668-8787-ca582a4209c5/scratchpad
RPT=$ROOT/experiments/report_final_20261105
rm -f "$M/README.md.bak"; rm -rf "$M/code" "$M/report" "$M/experiment_records" "$M/run_records" "$M/phase2_records" "$M/manifest" "$M/local_only"
mkdir -p "$M"/{code,report,experiment_records,run_records,phase2_records,manifest,local_only/sessions}
# report
rsync -a "$ROOT"/report_final/REPORT.md "$ROOT"/report_final/REPORT.zh-CN.md "$ROOT"/report_final/figures "$ROOT"/report_final/gen_report.py "$ROOT"/report_final/gen_figures.py "$ROOT"/report_final/build_materials.sh "$ROOT"/report_final/report_inputs.json "$M/report/"
mkdir -p "$M/report/src"; rsync -a "$ROOT"/report_final/src/tex2md.py "$ROOT"/report_final/src/REPORT.template.md "$ROOT"/report_final/src/REPORT.zh-CN.template.md "$ROOT"/report_final/src/tables_zh_strings.json "$ROOT"/report_final/src/check_translation.py "$M/report/src/" 2>/dev/null || true
# code: the current working tree (uncommitted changes included)
rsync -a --exclude __pycache__ --exclude '*.pyc' "$ROOT"/{train.py,predict.py,pp_eval.py,pp_study.py,make_submission.py,score_submission.py,prepare_course_data.py,requirements.txt,README.md,USAGE.md,setup.md,final.sh,run_all.sh,sarbench,tools,tests,docs} "$M/code/"
# keep_text: small text-like files, no predictions, data, upload receipts
FILTER=(--exclude __pycache__ --exclude '*.pt' --exclude '*.pth' --exclude '*.zip' --exclude '*.csv' --exclude '*.png' --exclude '*.idempotency_key' --exclude '*.response.json' --exclude 'predictions_*' --exclude 'detection.json' --exclude 'classification.csv' --max-size=3m)
rsync -a "${FILTER[@]}" --exclude raws/ --exclude ledgers/ --exclude code/ "$ROOT/experiments/" "$M/experiment_records/"
rsync -a "${FILTER[@]}" "$ROOT/.runtime/phase2/" "$M/phase2_records/"
B56=$(ls -d "$ROOT"/backups/56.49-*)
mkdir -p "$M/experiment_records/backup_56.49"; rsync -a "${FILTER[@]}" --exclude weights/ --exclude submission/ --exclude code/ --exclude 'REPORT*' --exclude 'report-page.html' "$B56/" "$M/experiment_records/backup_56.49/"
# run records: config, metrics and logs of every training run
cd "$RUNS"; find . -maxdepth 4 -type f \( -name metrics.json -o -name 'metrics_*.json' -o -name log.txt -o -name launch.log -o -name flags.txt -o -name README.md -o -name 'run.sh' -o -name 'leaderboard*.json' -o -name 'comparison*.json' -o -name 'status.json' -o -name 'args*.json' \) -size -3M ! -path '*_smoke/*' ! -path '*/code/*' ! -path '*/code_v2/*' ! -path '*/.pytest_cache/*' -print0 | rsync -a --from0 --files-from=- ./ "$M/run_records/"
[ -f events.log ] && cp events.log "$M/run_records/"; mkdir -p "$M/run_records/tracker"; [ -f tracker/notes.json ] && cp tracker/notes.json "$M/run_records/tracker/"
# documents exist in English (X.md) and Chinese (X.zh-CN.md); translations are kept outside the copied trees
rsync -a "$ROOT/report_final/translations/" "$M/"
# language switch on English documents copied from the working tree
python3 - "$M" <<'PY'
import sys
from pathlib import Path
m = Path(sys.argv[1])
for f in ['code/README.md', 'code/USAGE.md', 'code/setup.md', 'code/docs/DETAILS.md', 'experiment_records/bridge_followup/README.md', 'run_records/BP_queue/followup_20261028/README.md', 'phase2_records/PLAN.md']:
    p = m / f; L = p.read_text().split('\n'); i = next(k for k, l in enumerate(L) if l.startswith('# '))
    L[i + 1:i + 1] = ['', f'**English** · [中文]({p.stem}.zh-CN.md)']; p.write_text('\n'.join(L))
PY
# old report links of the 56.49 backup now point to the current report
sed -i 's#(REPORT.md)#(../../report/REPORT.md)#g' "$M/experiment_records/backup_56.49/README.md"; sed -i 's#(REPORT.zh-CN.md)#(../../report/REPORT.zh-CN.md)#g' "$M/experiment_records/backup_56.49/README.zh-CN.md"
# drop any json that holds per-image boxes (predictions) or that names labelled images (data)
cd "$M"; grep -rlE '"bbox"|"image_id"' experiment_records run_records phase2_records --include='*.json' --include='*.jsonl' 2>/dev/null | while read -r f; do rm -f "$f"; echo "$f"; done > manifest/removed_prediction_files.txt
find . -path ./local_only -prune -o -name 'vallike_split.json' -print | xargs -r rm -f
# local only (not pushed)
cp -al "$ROOT/submissions" local_only/submissions
cp -al "$ROOT/experiments" local_only/experiments
mkdir -p local_only/runtime; cp -al "$ROOT/.runtime/avg" local_only/runtime/avg; cp -al "$ROOT/.runtime/phase2" local_only/runtime/phase2
cp -al "$B56" local_only/backup_56.49
ln -s "$RUNS" local_only/cs701_runs
cp -a "$RPT/ledgers" local_only/ledgers
cp -a "$SP/condensed" "$SP/condensed_index.tsv" local_only/sessions/ 2>/dev/null || true
echo 'local_only/' > .gitignore
# manifest of what is not pushed
( cd "$ROOT/submissions"; find . -maxdepth 1 -type f -name '*.zip' -print0 | sort -z | xargs -0 sha256sum ) > manifest/submissions.sha256
( cd "$ROOT/.runtime/avg"; sha256sum *.pt ) > manifest/averaged_weights.sha256
( cd "$RUNS"; find . -type f \( -name '*.pt' -o -name '*.pth' \) -printf '%s\tcs701_runs/%P\n' ; cd "$ROOT"; find experiments .runtime backups -type f -size +3M -printf '%s\t%p\n' ) | sort -k2 > manifest/large_files_not_pushed.tsv
python3 "$RPT/progress.py" > manifest/training_status.txt
cp "$RPT/results/leaderboard_phase2.json" manifest/leaderboard_phase2_2026-11-05.json
# other people and other projects do not belong in a shared repository
grep -rIlE 'hegeng|SecureVibeGym' . --exclude-dir=local_only 2>/dev/null | while read -r f; do sed -i -E 's/hegeng/another user/g; s/SecureVibeGym/another project/g' "$f"; done
du -sh --exclude=local_only .; find . -path ./local_only -prune -o -type f -print | wc -l; find . -path ./local_only -prune -o -type f -size +20M -print
