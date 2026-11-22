# SC2 backbone candidates and course-compliance check

**English** · [中文](README.zh-CN.md)

Check date: 2026-10-31 (Singapore time). Subject: `51.86-full150ep-e130-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2`.

**Conclusion: alternatives exist whose source and parameter budget satisfy the public rules, but no ready-made upgrade was found with a same-config SAR experiment proving that it must beat SC2.** The genuinely newer candidates of the same scale are LingBot-Vision-B/16 (2026-07) and TIPSv2-B/14 (2026-04); the timm fixed-512 version of RADIOv3-B and DINOv3 ConvNeXt-B suit comparisons of multi-teacher features and of hierarchical structure respectively.

This round is research on public material plus a CPU/meta parameter audit; no large weights were downloaded, and no images were run, no training done, no GPU used and nothing submitted. Structural counts cannot replace verification of weight loading, forward/gradient and accuracy. The existing training code and queue were not changed.

## Official rules

Verified from the user-specified site: the [official Terms](https://web-production-c241e.up.railway.app/docs#terms) and the [public documentation asset](https://web-production-c241e.up.railway.app/assets/Docs-DK1tB6x4.js) they explicitly link. Anonymous access to `/team` returned only the public page shell; no private team account content was viewed.

- Generic pretraining is allowed; examples include ImageNet, CLIP, TerraMind and other EO foundation models, and the specific source must be reported.
- Forbidden: extra SAR data and detection/classification models trained on it. The rules explicitly distinguish foundation models pretrained on general satellite archives, so remote-sensing or Sentinel-1 pretraining cannot be excluded across the board.
- All weights of the final prediction model ≤130M, frozen ones included; all weights updated in training ≤40M, heads included. Full fine-tuning first and freezing later does not avoid the count.
- One model, one input per image, size ≤512×512; ensembles, TTA, tiling and oversize upscaling are forbidden. The official val/test are for prediction only.
- The report must disclose the model source, training-time teachers, parameters and all tuning compute cost, local splits, seeds and ablations; gains below about 2 Δm need a multi-seed mean and standard deviation.

The public terms do not define buffers item by item. The local SC2 check is more conservative in also counting buffers; learned weights are not exempt because they are registered as buffers. The buffers accounted for in the main candidates do not affect passing the budget.

## Parameter accounting

Layers and tensor shapes are built on the PyTorch meta device and `numel` is summed, without allocating actual weights. The rebuilt SC2 total matches the backup's `run/metrics.json` exactly. SC2's original gray backbone is 85,248,000 and its LoRA80 is 11,796,480; head/neck/RPN total 27,821,792, including three Cascade stages of width 1536.

| Specific option | Bare backbone | Total with head | Trainable | Accounting note |
|---|---:|---:|---:|---|
| SC2 DINOv3-B, gray, r80 | 85,248,000 | 124,866,272 | 39,656,672 | rebuilt from the original snapshot, plus 16 buffer elements |
| LingBot-B, gray, r80 | 85,276,416 | 124,894,688 | 39,656,672 | official constructor + original head, plus 27,664 buffer elements |
| LingBot-B, RGB, r80 | 85,669,632 | 125,287,904 | 39,656,672 | keeps the original RGB patch structure |
| TIPSv2-B/14, RGB, r80 | 86,316,288 | 125,934,560* | 39,656,672* | vision tower only, 32×32 position table |
| RADIOv3-B, timm fixed 512, RGB, r80 | 86,456,064 | 126,074,336* | 39,655,136* | official converted architecture |
| DINOv3 ConvNeXt-B, gray, MLP r80 | 87,560,320 | 127,116,192 | 39,594,016 | swapped to the existing hierarchical FPN, keeps Cascade1536 |
| DINOv3 ConvNeXt-B, gray, MLP r64 | 87,560,320 | 124,228,512 | 36,706,336 | as above, smaller adapter capacity |
| ConvNeXtV2-B, gray, MLP r80 | 87,686,656 | 127,242,528 | 39,594,016 | as above, GRN stays frozen |

`*`: backbone/adapter counted, plus SC2 heads of the same parameter count; end-to-end integration is not complete. The patch14 coordinate adaptation of TIPS may change the final neck design, so this number cannot be called a verified runnable model. All options keep the LoRA parameters unmerged. RGB options can build three channels from the same gray image, still with one forward pass; pretrained normalization has to be handled with the checkpoint.

The neighbouring JSON files hold the [original-model and LingBot counts](vit_counts.json), [TIPS counts](tips_counts.json), [RADIO counts](radio_counts.json) and [ConvNeXt counts](convnext_counts.json).

## Main candidates

### LingBot-Vision-B/16: first choice for a same-structure comparison

The [official model and weights](https://github.com/Robbyant/lingbot-vision#-model-zoo) describe pretraining on about 161M images organized from generic web images, ImageNet, landmarks, street views and so on, with B distilled from its generic Giant teacher. This source category satisfies the course's generic-pretraining rule.

Same-protocol B-level comparison in [paper Table 5](https://arxiv.org/html/2607.05247v1#S5.T5): DINOv3→LingBot ImageNet linear 84.79→85.05, NYU RMSE 0.371→0.339, ADE mIoU 51.74→51.44, VOC 88.20→87.10. Newer and stronger on depth, slightly weaker on segmentation; no results on the current SAR detection.

768 dimensions, patch16 and 12 layers are closest to the existing structure. Dictionary output, gray input, gradient checkpointing and masked bias still need adapting. The official attention reads `qkv.in_features`, and the existing LoRA wrapper does not pass this attribute through, so replacing only the name will not run directly. What was audited is structural parameters, not a wired-up model.

### TIPSv2-B/14: new 2026 candidate

[Official Google repository](https://github.com/google-deepmind/tips), [Base weights](https://huggingface.co/google/tipsv2-b14), [paper](https://arxiv.org/html/2604.12012v1). Released on 2026-04-13, on generic WebLI image-text data and synthetic captions, about 116M images; using only the vision tower fits the generic-pretraining category. The original vision + text package is about 196M and cannot be put as a whole into the final prediction model.

The official README says 85.7M, but the v2 inference code explicitly has a 32×32 position table, and this RGB construction counts 86,316,288, matching the paper's about 86.3M. The paper's "wins four of six against DINOv3" is at L level and includes image-text tasks, and its ADE is still 51.4 vs 54.9, so stronger B-level SAR cannot be inferred.

Keeping patch14 allows a single 504×504 input, but the FPN, anchor and RoI scale relationships must be corrected; the existing stride16 default implementation cannot be copied. 448 is also compliant but shrinks the objects. Whether text alignment helps fixed-class SAR needs a local ablation.

### C-RADIOv3-B: usable once a specific checkpoint is fixed

The 128×128 position table of the [NVIDIA original](https://huggingface.co/nvidia/C-RADIOv3-B) takes 12,582,912 parameters, and the whole original module is 98,254,848, so it cannot be swapped in directly at the table's about 90M. Original RGB + r80 + original head is 137,873,120, over budget; r24 can bring it down to 129,615,584/31,397,600, but adapter capacity changes.

The [timm fixed-512 model card](https://huggingface.co/timm/vit_base_patch16_radio_v3.c_radio) states: the position table is resampled once to 32×32 by the original method, with no extra training; 512 output matches the original, and other resolutions may deviate. The source weight revision is `44653a0482cf460bb4f12595fc3cc3dfecc403d1`. This round verified the official description and code, and did not independently load the weights to verify numerical equivalence.

The fixed-512 version has 86,456,064 parameters; keeping RGB + r80 + original head the budget is 126.07M/39.66M. The data is NV-CC-Img-Text and the teachers are DFN CLIP, SigLIP2, DINOv2 and SAM, fitting the generic-pretraining category; final inference uses only the student vision tower and does not run these teachers. v3 pretraining was released in 2025; a new timm port does not equal a new generation of pretraining, and there is no evidence of a same-config SAR win.

### DINOv3 ConvNeXt-B: structural comparison

The [official model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md) states that it uses generic LVD-1689M. It provides 4/8/16/32 hierarchical features; swapped to the existing FPN with MLP two-layer LoRA80, the original Cascade1536 can be kept and the budget passes. Small-scale features are the motivation for the experiment, not evidence of a gain: official ADE is 46.3 for ConvNeXt-B and 51.8 for ViT-B.

The earlier project trained ConvNeXt-T with weak results, and B/S were only downloaded then; T's result cannot be extrapolated to say B has failed. Its method is the same generation as the current DINOv3.

## Other candidates and reasons for exclusion

| Family | Conclusion |
|---|---|
| [PE-Spatial-B16](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md) | Generic spatial distillation, RGB about 86.43M; 2025-07, earlier than DINOv3. Bare Parameter attention projections need dedicated LoRA adaptation; no same-config SAR advantage found. |
| [EVA-02-B/p16](https://github.com/baaivision/EVA/blob/master/EVA-02/det/README.md) | Official patch14→16 weights exist, about 86M, generic ImageNet pretraining. 2023 model; COCO 55.5 AP uses 1024 input and a different recipe, and cannot be transferred directly to this task. |
| [SAM2.1 Hiera-B+](https://github.com/facebookresearch/sam2#model-description) | 80.8M is the full SAM2, not the bare backbone; the image encoder can be extracted for research. The default 1024 predictor does not meet the size requirement, needs ≤512 and removal of the unused memory/prompt/mask components; 2024 model. |
| [ConvNeXtV2](https://github.com/facebookresearch/ConvNeXt-V2), [SwinV2](https://github.com/microsoft/Swin-Transformer), [VMamba](https://github.com/MzeroMiko/VMamba), [MambaVision](https://github.com/NVlabs/MambaVision), [InternImage](https://github.com/OpenGVLab/InternImage) | Generic pretraining exists and within-budget designs are possible; most methods are older. Operators, feature outputs and adapters need more changes, and original COCO scores are not equivalent to a SAR comparison at 512 with a 40M adapter budget. Extended structural counts are in the additional JSON. |
| [Franca-B](https://github.com/valeoai/Franca) | Method 2025-07, generic IN21K; same-protocol B-level performance in the LingBot paper does not support replacing DINOv3 first. |
| TerraMind, DOFA and other generic EO models | Cannot be excluded by a remote-sensing/SAR name; TerraMind is an example the terms explicitly allow. The current 8-bit high-resolution gray images have a domain gap from satellite-archive inputs; this project's early TerraMind results did not prove it better. |
| [DINOSAR](https://huggingface.co/YTang/DINOSAR) | Not put on the "certainly compliant" list. [UniSAR-7M](https://huggingface.co/datasets/YTang/UniSAR-7M#data-sources-and-composition) directly includes training images from SARDet-100K, ATRNet-STAR, FAIR-CSAR and others, some screened using annotations; the general satellite-archive foundation-model exception cannot simply be applied. |
| DINOv3-L/SAT, RADIOv4-SO400M/H, large PE models, SAM3 original encoder | For the public models involved, a single backbone is already clearly over budget; freezing/LoRA does not change the final total of weights. |
| "DINOv4" | No usable official Meta release found in this check; not listed as an executable candidate. |

## Recommended experiment order and limits of interpretation

First compare DINOv3-B with LingBot-B, with local split, head, schedule and post-processing fixed, focusing on mAP, AP75, small AP and weak-class AP. Choose the second group by question: TIPSv2 for a new pretraining method, RADIO fixed 512 for multi-teacher features, ConvNeXt-B for multi-scale structure. Train each option independently, to avoid changing several factors at once and attributing all the gain to the backbone.

Screen on the local holdout; local images the full-data model has already seen are not used to claim generalization improvement. Tiny gains need paired multi-seed rechecks. After finally loading specific weights and completing the interfaces, reconfirm all inference weights, all trained weights, input size, single forward pass and coordinate consistency. This round adds no new accuracy results.
