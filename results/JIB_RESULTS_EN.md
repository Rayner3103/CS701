# JIB final training results

Training completed on 31 October 2026 at 04:48 CST: 100 epochs, 49,900 optimizer updates and 26 planned validation evaluations. The supervisor exited with code 0. The selected epoch is 88, chosen by the highest internal Delta_m among the planned evaluated epochs.

| Checkpoint | Macro-F1 (%) | COCO mAP (%) | Internal Delta_m (%) |
|---|---:|---:|---:|
| Selected epoch 88 | 95.03 | 47.50 | +41.81 |
| Final epoch 100 | 95.09 | 47.44 | +41.73 |

These are fixed internal-holdout results, not official leaderboard scores. Delta_m (%) = 50 x [(macro-F1 - 0.9331)/0.9331 + (mAP - 0.2613)/0.2613]. Training used 7,984 images and validation used 1,408 images. Classification is macro-F1 over nine classes; detection is COCO mAP at IoU 0.50:0.95.

The model uses the earlier F2 DINOv3/r32 three-stage Cascade design with instance-balanced positive RoI sampling in all three stages. Native downscale-only inputs and valid-region geometry are applied. No earlier local-model code/weights or excluded SC2 configuration is imported. The sampler adds no parameters or inference passes. Total parameters: 109,014,752; trainable parameters: 23,805,152.

The run includes recovery from complete epoch boundaries after interruptions; incomplete epochs were replayed with model, optimizer, scheduler and RNG restored. The source and 100-epoch schedule were unchanged.

The selected checkpoint has AP50 71.25% and AP75 50.36%. The weakest detection classes remain bridge (AP 1.91%) and playground (AP 15.77%). These class-level results describe the selected internal checkpoint and do not establish the cause of the errors.

A matched native F2 control is still required to isolate the sampling effect. One seed and a different evaluation split cannot establish superiority over a teammate or a leaderboard submission. No official prediction upload or full-data final fit has occurred.

Selected weight: `D:/CS701_junkai_ib/runs/JIB_seed701/best_epochs/epoch_0088.pt`.
SHA-256: `1d7ce4ebaa6723aaf99e5d72a053b4480ea82efbff486f68ddd5a224d6f3dbf7`.
Evidence: `D:/CS701_junkai_ib/completion_verification_20261031.json`.

## All planned evaluated epochs

| Epoch | Macro-F1 (%) | COCO mAP (%) | Internal Delta_m (%) |
|---:|---:|---:|---:|
| 1 | 83.89 | 1.23 | -52.69 |
| 4 | 91.07 | 13.89 | -24.63 |
| 8 | 93.54 | 21.51 | -8.72 |
| 12 | 91.12 | 23.88 | -5.47 |
| 16 | 93.16 | 27.22 | +2.01 |
| 20 | 93.38 | 30.49 | +8.39 |
| 24 | 94.43 | 33.74 | +15.16 |
| 28 | 93.01 | 32.07 | +11.21 |
| 32 | 94.79 | 37.92 | +23.35 |
| 36 | 95.38 | 39.78 | +27.24 |
| 40 | 95.21 | 40.30 | +28.12 |
| 44 | 95.32 | 42.05 | +31.53 |
| 48 | 95.37 | 42.95 | +33.29 |
| 52 | 94.89 | 44.10 | +35.23 |
| 56 | 95.10 | 44.61 | +36.32 |
| 60 | 95.11 | 45.48 | +37.98 |
| 64 | 94.92 | 45.72 | +38.35 |
| 68 | 94.72 | 46.05 | +38.87 |
| 72 | 95.08 | 46.86 | +40.62 |
| 76 | 94.91 | 47.04 | +40.88 |
| 80 | 94.91 | 47.39 | +41.54 |
| 84 | 94.96 | 47.31 | +41.42 |
| 88 | 95.03 | 47.50 | +41.81 |
| 92 | 94.97 | 47.47 | +41.72 |
| 96 | 95.03 | 47.49 | +41.78 |
| 100 | 95.09 | 47.44 | +41.73 |

Next work: verify prediction/export with the selected trained checkpoint, prepare an auditable delivery package, and decide whether to run a matched control or a full-data fit. Competition submission remains a separate action.
