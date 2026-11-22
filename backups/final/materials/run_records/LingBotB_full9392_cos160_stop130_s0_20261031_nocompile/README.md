# LingBot-Vision-Base replacing the BP7 backbone

**English** · [中文](README.zh-CN.md)

2026-10-31 user authorization: "after this one finishes, try swapping in LingBot-B". After the current BP7 completes e130 and exports its submission, one LingBot-B comparison runs automatically. A short dual-GPU smoke test runs first and formal training follows only if it passes; failure keeps the state and stops.

## Fixed configuration

- LingBot-Vision-Base ViT-B/16, 768 dimensions, 12 layers, replacing DINOv3-B.
- Keeps the original BP7 LoRA80/alpha160, Cascade1536, SpatialPrior64, NWD8, SizeIoU, aligned RoI, flip-bias−0.3 and anchors; no context.
- All 9392 images of the official train. Actual paths of local train 7512+val 940+test 940 are all verified to point to the official train; official val 1426 is inference only.
- Head LR 1e-4, Backbone LR 3e-4, warmup 1 epoch, weight decay 0.05, seed0, global batch16.
- 160-epoch cosine, stopping at e130, 76310 update steps. A full checkpoint is saved every epoch and inference weights every 10 epochs; it does not continue to 160 automatically.
- Dual-GPU, CPU0–3, nice10, 2 workers/compute threads per GPU.
- The gray patch kernel is the sum of the RGB weights, keeping the same SAR normalization. LingBot pretraining's random RoPE coordinate scaling is turned off and the RoPE position encoding kept, consistent with the existing BP7 downstream augmentation recipe.
- e130 automatically exports submissions/LingBotB_full_cos160_e130.zip, with the historical post-processing and shift−0.2; a single 512 forward per image. Validates coverage of 1426 images, format, finite boxes, bounds, 100 boxes per image, ZIP integrity and the 25MB limit, and saves SHA256. Submitted manually by the user.

## Fixed sources and verification

Official code https://github.com/Robbyant/lingbot-vision pinned at commit 151e46321bae4399f8568829f190c7bdec216b49. The original files and LICENSE/LEGAL/NOTICE/UPSTREAM are kept in sarbench/vendor/lingbot_vision; the adaptation is separately in sarbench/lingbot.py.

Official Base weights https://huggingface.co/robbyant/lingbot-vision-vit-base pinned at revision f606f8c6c4002234ea68038f4d7c7cf57da96dfa. model.pt is 342852282 bytes, SHA256 783dfb59014c34e9f0013db60bf6f5cc3c6b0604eb9528fb5df65e80769c5825. Download verification is in pretrained.json.

The public source is generic-image self-supervised pretraining, with Base distilled from the generic Giant; only the Base vision backbone is used. Per the public course terms https://web-production-c241e.up.railway.app/docs#terms generic pretraining is allowed. Measured: total parameters 125105376, trainable 39867360, floating-point buffers 27664, parameters plus buffers 125133040, within the 130M/40M budget.

CPU check on real weights: strict loading, forward error between gray-sum and RGB repeated channels below 5e-6, finite gradients on 48 LoRA layers, gradient checkpoint forward/gradient consistency, Dynamo single-block fullgraph+checkpoint pass. CPU cannot replace the CUDA/Inductor dual-GPU smoke test, which is scheduled after the current BP7 completes.

## Queue and interpretation

Predecessor: /mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full9392_cos160_stop130_s0_20261031.

New queue: /mnt/hdd/junkaichen/CS701/cs701_runs/LingBotB_full9392_cos160_stop130_s0_20261031. Controller tools/run_lingbot_full_stop130.py waits for the predecessor's e130 weights and submission validation to pass and for the lock to be released. The 18 queue CPU tests pass.

The current BP7 frozen snapshot is not modified. There is no evidence yet of the effect on this SAR task. Local val/test after full-data training have taken part in training and are not used to claim generalization improvement; the comparison is with the official e130 result of BP7 with the same recipe. Small single-seed differences cannot be taken as a stable improvement. Smoke test, formal training and search costs are fully recorded.

Recovery: the original CUDA compiled smoke failed before its first optimizer update. This suite disables block compilation only and repeats the complete smoke gate. Original failure and source snapshot are preserved; see recovery.json.
