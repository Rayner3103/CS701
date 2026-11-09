# junkai_os3: Junkai's best-model code + 3x small-object oversampling

Copied from `origin/junkai` (`18018bb`), `backups/final/materials/code/` (the code snapshot of the 56.49 / 61.25 model), minus `docs/` images.
Run from the repo with `gpuClusterRun/junkai_best_os3.sh` (see its header). Changes to Junkai's code, nothing else was edited:

| File | Change |
|---|---|
| `sarbench/data.py` | `SARMultiTask(oversample_small=N, small_px=32)`: in training, every image with a box of area < small_px^2 on the 512 input (image stretched to 512 x 512, as in the model's input) appears N times, originals kept, val/test never. Same definition as the rayner branch's `augment_small_objects.py --oversample N --paste-prob 0` (os3), applied in the loader instead of on a YOLO copy of the data. |
| `train.py` | `--oversample-small N` (default 1 = off) and `--small-px`; the training loader prints the repeated-image count. Junkai's class-weighted `--oversample` is untouched. |
| `tools/make_vallike_split.py` | new: rebuilds `vallike` (992) / `trainvl` (8,400) from the per-(class, size) cell targets of REPORT.md. His builder was never pushed, so the images differ, the mix is the same (`trainvl` gives the 525 steps per epoch of the report). |
| `tools/local_eval.py` | new: FINAL post-processing + lattice snapping + COCO AP + macro-F1 + delta_m on a raw dump (self-contained version of his `proxy_eval.py` / `boot_eval.py` / `snap_eval.py` / `lattice_eval.py`, which hard-code his paths). |
| `tests/test_oversample_small.py` | new tests for the three items above. |

Measured on this data: os3 turns the 8,400 `trainvl` images into 17,002 training items per epoch (4,301 images have a small box): 1,063 steps instead of 525.
