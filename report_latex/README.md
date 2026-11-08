# CS701 Team 7 final report (CVPR LaTeX)

Build: `latexmk -pdf main.tex` (6 pages including references; `main.pdf` is the compiled copy).

- `cvpr.sty` / `ieeenat_fullname.bst` come from the CVPR 2026 author kit (branch `jinling`), with `\confYear` set to 2027. Replace `cvpr.sty` with the course's CVPR 2027 kit if it differs, then rebuild.
- Figures: `box_sizes`, `ap_vs_iou`, `per_class_ap`, `near_duplicates` come from `backups/final/materials/report/figures` (branch `junkai`); `cnn_and_unfreeze.pdf`, `patchgrid.png` and `bridge_gt.png` are new; the pipeline figure is TikZ in `main.tex`.
- Content: Junkai's final REPORT.md (8 Nov) plus Luat's CNN detector line and full-block screens (Secs. 3.2, 3.3, 4.3, 4.4, 4.6).
