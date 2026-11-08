# CS701 Team 7: Final Materials

**English** · [中文](README.zh-CN.md)

The report, code and experiment records as of 8 November 2026. All 30 report-training tasks finished successfully at 03:24 SGT; the report includes the three-run results. Items still marked TBD require records that are not available. Documents come in two versions: `X.md` is English and `X.zh-CN.md` is Chinese.

**Start here: [report/REPORT.md](report/REPORT.md)** (the report with charts and the complete experiment log).

## Contents

| Path | Contents | Pushed |
|---|---|---|
| `report/` | `REPORT.md`, `REPORT.zh-CN.md`, charts in `figures/`, and the scripts that generate them | Yes |
| `code/` | Code snapshot of the final model and report runs | Yes |
| `experiment_records/` | Notes, configs, scripts and result summaries of each experiment; `backup_56.49/` holds reproduction notes and run records of the 56.49 entry | Yes |
| `run_records/` | Config, metrics and logs of each training run | Yes |
| `phase2_records/` | Phase 2 plan, queue and evaluation scripts, TWIN1 verdict | Yes |
| `manifest/` | Lists and checksums of large files not pushed, training status, Phase 2 board snapshot | Yes |
| `local_only/` | Weights, raw outputs, submission files, session records, experiment ledgers (on the server only) | No |

## Phase 2 result

Δm **61.25**, macro-F1 96.96, mAP 55.74, AP50 75.19, posted 3 November 15:37 SGT. Rank 3 of 9 on the public board at 12:21 SGT on 8 November; the board was not locked and closes at 23:59 SGT that day. This is not a final ranking. Snapshot: `manifest/leaderboard_phase2_2026-11-08.json`; retrieval time: `manifest/leaderboard_snapshot_meta.json`.

## Completion and remaining items

| Item | Status |
|---|---|
| File and number of Phase 2 submissions | TBD: the last one is recalled as `PHASE2_shot1_FINAL_avg100-180x9_agn_mk100_SNAP.zip`; the board shows no file names |
| Other authors | TBD |
| Full-length seeds 1 and 2 | Complete, 180 epochs each; together with seed 0, three full-length local runs; see `manifest/training_status.txt` |
| Leave-one-out ablations (8 variants, 3 seeds) | Complete: 27 runs including three paired baselines, 36 epochs each; Table 2 |
| Local reference run | Complete: 24 epochs; F1 92.31, mAP 25.66, AP50 54.81 with default evaluation |
| Project GPU-hour total | About 514 = historical about 388 + report queue 126.31; queue time includes prediction and averaging, excludes teammates' RTX 3080 runs; `manifest/report_compute.json` |
| Report format and due date | TBD: "CVPR template, at most 6 pages" comes only from a team draft; no page limit is applied for now. A LaTeX version is on hold |

## Not pushed, and why

The repository is public, the Terms do not allow sharing data or predictions outside the team, and Phase 2 is still open.

- **Predictions and data:** submission zips, JSON files with per-image boxes, labelled split files. For the same reason the report's charts contain no SAR images.
- **Weights and raw outputs:** single files reach over 600 MB, above GitHub's limit. See `manifest/large_files_not_pushed.tsv` and `manifest/*.sha256`.
- **Session records and experiment ledgers:** they hold chat text and internal discussion.
- **Old reports of the 56.49 backup:** some content is out of date; use `report/REPORT.md`.

Not translated: logs, data files, and third-party `LICENSE`, `NOTICE` and `LEGAL` files.

## Rebuild

Regenerate the report from this materials directory (using the report Python environment):

```bash
python report/gen_figures.py
python report/gen_report.py
```

Rebuild the entire bundle on the original server, where the source experiments are available:

```bash
report_final/build_materials.sh   # from the repository root
```
