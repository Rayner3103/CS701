# Comparison of three submissions

**English** · [中文](README.zh-CN.md)

All with 7512 training images, original learning rate, 160-epoch cosine schedule, single 512 input, unified post-processing shift−0.2.

|Label|Model/checkpoint|Purpose|
|---|---|---|
|A|Original BP7 e150|late baseline|
|B|Context BP7 e150|compare the architecture with A at the same epoch|
|C|Context BP7 e130|compare with the already submitted original BP7 e130 (47.12), testing earlier stopping|

All three pass the format, 1426-image coverage, bounds, count, CRC and size checks. Not uploaded. results.json holds paths and hashes, and each subdirectory keeps the command, sources, raw outputs and validation records. C is e130 on the 160-cos trajectory, not a 130-cos training.


The user asked to simplify file names: A→submissions/a.zip, B→submissions/b.zip, C→submissions/c.zip. The ZIP and SHA256 files were renamed directly; validation records were updated accordingly, and file contents are unchanged.


A official feedback (2026-10-31 screenshot): Δm 46.92, F1 94.48%, mAP 50.32%, accuracy 94.88%, AP50 71.65%. Against the 47.12 of e130 on the same 160-cos trajectory, Δm −0.20; the screenshot's +1.09 is relative to the posted entry, not to e130. B/C feedback pending.


B official feedback: Δm 46.89, F1 94.89%, mAP 50.19%, accuracy 95.16%, AP50 72.07%. At the same e150 relative to A: Δm −0.03, F1 +0.41pp, mAP −0.13pp, accuracy +0.28pp, AP50 +0.42pp. Overall nearly flat; context improves classification but brings no detection mAP gain; not yet a conclusion of stable gain or harm.


C official feedback: Δm 46.93, F1 94.78%, mAP 50.24%, accuracy 95.09%, AP50 72.33%. Relative to no-context at the same e130: Δm −0.19, F1 +0.20pp, mAP −0.17pp, accuracy +0.14pp, AP50 +0.71pp.

The three Δm: A 46.92/B 46.89/C 46.93, a range of only 0.04. Neither same-epoch comparison shows an overall gain from context; no-context e130→e150 is 47.12→46.92, and context e130→e150→e160 is 46.93→46.89→46.80. The current trajectory gained nothing online from later checkpoints, but tiny single-seed differences do not prove that longer training is generally ineffective. The recommendation is to stop spending submission quota around these neighbouring checkpoints.
