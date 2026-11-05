# Paired diagnosis on the second local holdout

**English** · [中文](README.zh-CN.md)

Original BP7 e150 and context e150 are both train 7512, seed0, 160cos, unified post-processing, local test 940. This holdout has been used in the past and is not a new blind test. This round compares only the two fixed sets of weights and does no tuning on it.

Original model mAP 53.2262%, F1 94.5725%, local Δm 65.8027; context mAP 53.0473%, F1 95.0628%, Δm 65.6865. Differences: mAP −0.1789pp, F1 +0.4903pp, Δm −0.1162, overall nearly flat. The mAP +0.966pp on the original val did not repeat on this split; it cannot be claimed that context stably improves or is necessarily harmful.

The user then asked for 3 official submissions for feedback, recorded in the neighbouring context_submission_trio_20261031 directory.
