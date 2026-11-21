# CS701 Team 7 final report (CVPR LaTeX)

Build: `latexmk -pdf main.tex` (6 pages including references; `main.pdf` is the compiled copy).

- `cvpr.sty` / `ieeenat_fullname.bst` come from the CVPR 2026 author kit (branch `jinling`), with `\confYear` set to 2027. Replace `cvpr.sty` with the course's CVPR 2027 kit if it differs, then rebuild.
- Figures used: `box_sizes`, `ap_vs_iou` (from `backups/final/materials/report/figures`, branch `junkai`); `cnn_and_unfreeze.pdf` and `leakage.png` are new; the pipeline figure is TikZ in `main.tex`. `per_class_ap.png`, `near_duplicates.png`, `patchgrid.png` and `bridge_gt.png` are no longer referenced (kept for the slides).
- Content: Junkai's final REPORT.md (8 Nov), Luat's CNN detector line and full-block screens (Secs. 3.2, 3.3, 4.3), and the train-test leakage audit (Sec. 5) added after the organisers released the val/test labels on 13 Nov.

## Leakage audit (Sec. 5) — `leakage_analysis/`

Inputs: the course data (`train/`, `val/`) and, from `hf download doem1997/cs701-sar-course-data`, `test.tar.gz`, `val_labels.csv`, `val_instances.json` (+ test files). Validation submissions: `backups/*/submission/*_val_*.zip` on this branch and the CNN-line zips in Luat's `artifacts/`.

1. `emb.py TEST_DIR nn.npz DATA_ROOT` — frozen DINOv3 ViT-B/16 (timm `vit_base_patch16_dinov3.lvd1689m`, 256 px) global features; nearest-train cosine of every val/test image, plain and max over the 8 flips/90° rotations of the query (GPU, a few minutes).
2. `pairs.py DIR` — pixel correlation (64×64) of each best pair after undoing the flip/rotation → `pix.npz`; prints tier shares and writes a contact sheet used to set the 0.94 threshold.
3. `score_bands.py DIR HF_DIR zip...` — macro-F1, COCO mAP and Δm (val reference 93.31 / 26.13) per tier: exact (pixel NCC ≥ 0.8), near (cosine ≥ 0.94, not exact), none (< 0.94). Overall Δm reproduces the leaderboard to 0.01.
4. `boot_copy.py DIR HF_DIR TRAIN_DIR NAME=zip...` — paired image bootstrap of model differences per tier (env `PAIRS=F3-SC2`, `GROUPS=far,nonexact`, `REPS`), and the 1-NN copy baseline (label and flip/rotation-mapped boxes of the matched training image).
5. `fig_leak.py DIR out.png 2` — Figure 4.

`results/` holds the outputs used in the paper (`bands_all.json`, `bands_f3e130.json`, bootstrap logs, `nn.npz`, `pix.npz`). Note: `pycocotools`' `COCOeval.accumulate(p)` indexes image subsets wrongly, so subset mAP is computed by rebuilding `evalImgs` (checked against a fresh `COCOeval` on the subset).
6. `shiftcheck.py DIR` — shift/scale-tolerant template matching (OpenCV, centre and quadrant crops, scales 0.8/1/1.25) of each near/none pair → `shift.npz`; prints the verified share per tier and a random-pair control (0.2% false positives). `score_strict.py` re-scores the none tier without the pairs it verifies (`bands_strict.json`).

Figure 4 is `fig_leak.py DIR figures/leakage_pairs.png 2 only` (image row only); `leakage.png` is the older version with the bar chart.

## Reproducing the final entry

Training, prediction, checkpoint averaging and snapping code is on branch `junkai` under `backups/final/materials/code` (see its README and `backups/final/materials/experiment_records/backup_56.49/make_submission.sh`). The final checkpoint and the submitted zips are not in git.
