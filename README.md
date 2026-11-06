# CS701 SAR joint classification and detection — client handover

Delivery date: 1 November 2026. This package contains the exact Python training/inference source snapshot used for the completed JIB_seed701 experiment. It is an attributed adaptation of Junkai's earlier F2 implementation, not a wholly independent architecture. The larger detection-head redesign discussed later is NOT implemented here.

## Start here

1. Read `docs/CHANGES_EN.md` for the precise differences and attribution.
2. Read `docs/QUICKSTART_EN.md` for installation, data preparation, training and inference.
3. Read `results/JIB_RESULTS_EN.md` for all validation results and limitations.
4. Read `docs/FILE_MAP_EN.md` to find each implementation.

Best internally evaluated checkpoint: epoch 88; macro-F1 **95.03%**, COCO mAP **47.50%**, internal Delta_m **+41.81%**. Training completed 100 epochs / 49,900 optimizer updates. This is a one-seed internal 7,984/1,408 split experiment, not an official leaderboard result.

The code-only ZIP excludes weights and course images. The companion with-model ZIP includes `weights/JIB_epoch88.pt`, the exact full selected checkpoint. Both use identical code. Generic DINOv3 initialization and raw course images must be obtained separately for retraining. No competition upload is performed by these commands.

All model Python files retain their frozen training hashes. Delivery instructions and test copies are added outside the model implementation. Historical upstream entry points remain for provenance: use `run_project.py` and `predict_native.py`, not upstream `train.py`/`predict.py` main commands or the hardcoded operational qualification scripts.

## GitHub branch layout

This branch contains the Jinling handover under `code/`, `configs/`, `results/`, `evidence/`, and the English handover documents in `docs/`. Use `code/run_project.py` and `code/predict_native.py`; the original repository files outside `code/` remain historical upstream material. This run trained exactly **1 seed (701)**. The weights are supplied separately as `JIB_epoch88.pt`; they are not stored in this Git branch. The ZIP-package references above refer to the separately delivered archives.
