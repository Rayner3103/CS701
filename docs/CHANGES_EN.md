# What changed relative to Junkai F2?

## Retained from the earlier F2 implementation

DINOv3 ViT-B/16; rank-32 attention/MLP LoRA and trainable norms; the simple feature pyramid; small, offset anchors; RPN NWD matching (C=8); aligned RoI extraction; A4 size-aware matching; three-stage class-agnostic Cascade; and the core objectness/soft-NMS/class-fusion postprocessing. These are upstream contributions, not ours. The excluded SC2 / 51.86 parameter-expansion recipe is not used.

## Main experimental method change: instance-balanced positive RoI sampling

Source: `code/sarbench/instance_sampling.py`; activated in `code/train.py`.

The original positive sampler randomly samples proposals from one pool. Objects with many matching proposals can occupy more of the positive budget. We group positive proposals by matched ground-truth object, randomize within each group and randomize group priority, then select proposals in round-robin order across objects. Negatives remain uniformly sampled. The original budget of 512 RoIs/image and a maximum of 128 positives is retained in each of the three Cascade stages. Sorting the sampled original indices preserves appended-GT bookkeeping.

This changes the training samples, adds no learned parameters and adds no inference pass. Instance balancing is an established detection idea; we claim the implementation and evaluation in this model, not invention of a new sampling algorithm. Reference: https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/instance_balanced_pos_sampler.py

## Input geometry adaptation

Sources: `code/sarbench/data.py` and `code/sarbench/model.py`.

Replace unconditional 512-square resizing with aspect-preserving downscaling only when needed, followed by padding. A 256-pixel image is not enlarged. Track valid image sizes and actual per-axis resize scales; pool classification features over valid regions; clip proposals and Cascade boxes to valid sizes; infer RoI feature strides from the padded canvas; map boxes back to original coordinates; and apply flip-bias correction in original-pixel units. The same preprocessing is used for labeled and unlabeled images. The coordinate postprocessing shift is 0 for this adaptation. These are implementation/input adaptations, not a demonstrated accuracy gain.

## Training and evaluation infrastructure

New standalone runner supports effective batch16 using batch4 x accumulation4, full optimizer/scheduler/RNG recovery, fixed split/source/data fingerprints, scheduled validation, and checkpoint selection by the course Delta_m formula. The native prediction entry reads the full checkpoint and exports original-coordinate results. Generic pretrained DINOv3 weights are used; no teammate-trained checkpoint or prior rejected local model is imported.

## Limits of the claim

The architecture remains F2-derived. A matched native-F2 control has not been run, so +41.81% internal Delta_m does not isolate the sampler's effect or prove superiority over Junkai. The official score is unknown. The previously discussed separate classification/convolutional localization head and multi-scale redesign are not present in this package.

Upstream repository: https://github.com/Rayner3103/CS701
Snapshot commit: 8445d584257ea8af99734f4ff18b890dcd1ba749. Original file hashes are preserved in `UPSTREAM_PROVENANCE.json`. No new license is asserted for the upstream code; upstream attribution is retained.
