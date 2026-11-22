# CS701 autoresearch: rules, existing evidence and next-round candidates

**English** · [中文](README.zh-CN.md)

Research date: 2026-10-31, Singapore time. The user allowed any rule-compliant improvement to be considered; this round changed no training queue, training weights or submission settings, and started no extra GPU jobs. Completed: an official-rules check, an audit of past experiments, training-box statistics, and a CPU TIDE diagnosis on existing predictions.

## Conclusion

The best-supported direction at present is **reducing missed detections in the weak classes and improving localization at strict IoU**. First validate BP7 combined with classification-only context, and diagnosis-driven positive-sample matching / box regression; continuing a blind sweep of peak LR, simply raising the RPN proposal count, or tuning NMS has weaker evidence. New designs are tested on the local training split first; official submissions are not used as a search loop.

## Rule boundaries (checked live)

Based on the [official Docs](https://web-production-c241e.up.railway.app/docs) and that page's public component Docs-DK1tB6x4.js as of 2026-10-31: one model, one input per image, size at most 512×512; total parameters ≤130M, parameters updated in training ≤40M. Forbidden: prediction ensembles, TTA, tiling, extra SAR data, and models trained on extra SAR detection/classification data. The official val/test may only be predicted; they may not be used for pseudo-label training, adaptation or manual labelling. The local 940/940 come from the official train and may be included in the final full-data training.

Generic pretraining may be used and its source must be reported. Training-only teachers must be mentioned in the report, but the teacher's source and the training-parameter count must still meet the constraints; this is not read as approval for any teacher. EMA/SWA/weight averaging has not been ruled on separately; it is listed as needing confirmation for now, and no averaged-weight submission is generated automatically. Weights that were updated before being frozen must still be disclosed truthfully; the count of trainable parameters at any one step is not used to get around the limit.

The current model has 125,076,960 total parameters and 39,867,360 trained parameters, leaving only 132,640 of trainable-parameter room.

Phase 1 reference: F1=93.31%, mAP=26.13%. By the official formula, each 1 percentage point of mAP adds about 1.9135 Δm and each 1 percentage point of F1 adds about 0.5358 Δm, so per percentage point mAP carries about 3.57 times the weight. A small F1 gain must not be bought with a larger mAP loss.

Phase 1 allows 5 submissions per day; Phase 2 allows 2 in total. Phase 1 closes on 1 November 23:59 SGT; Phase 2 runs from 2 November 00:01 to 8 November 23:59. The publicly posted score is the one selected; with nothing posted, the last successful submission of the phase is taken, and the best of all private history is not selected automatically. For gains below about 2 Δm, the rules require a mean and standard deviation across seeds; the current +1.29 cannot be called a stable gain. The [current official page](https://web-production-c241e.up.railway.app/docs) prevails.

## Local evidence

Full numbers and source-file SHA256 are in [data_audit.json](data_audit.json). The baseline is `BP7_LR130_local_s0_20261029_v2/lr1x`, train 7512, val 940, seed0; the class APs below use the unified submission post-processing.

| Class | AP | Median sqrt(area) of training boxes, at 512 scale | Share below 16 px |
|---|---:|---:|---:|
| bridge |4.78%|9.91|82.8%|
| playground |22.94%|14.88|64.0%|
| ship |39.14%|55.71|0.5%|
| wind_turbine |72.98%|8.16|94.2%|

Of the 48 image-classification errors, 24 are bridge↔playground. The weak classes are not simply short of samples: bridge has 999 training images and playground 939, while car has only 355 yet AP 62.75%. Bridges with a long-to-short side ratio >3 are only 3.3%, playgrounds only 0.1%; extending extreme anchor aspect ratios is not the first choice for them. wind_turbine is smaller yet clearly better, so size is not the only cause either.

### New CPU error breakdown

Uses the [TIDE paper](https://arxiv.org/abs/2008.08115) and the [official implementation](https://github.com/dbolya/tide) 1.0.1 to analyze the existing `predictions_val.json`. This is the native training-time post-processing, **not the submission post-processing**, with no extra forward pass and no official val labels. AP50=73.245955 and AP75=56.319134 agree with the original COCO evaluation to numerical precision.

| Positive-match IoU | AP | Oracle dAP of missed detections (Miss) | Oracle dAP of localization (Loc) | Oracle dAP of background (Bkg) |
|---|---:|---:|---:|---:|
|0.50|73.246|14.736|3.461|1.341|
|0.75|56.319|9.323|20.437|0.741|

The results support investigating missed detections and fine localization first. Oracle dAP is the effect of ideally correcting one kind of error; the values cannot be added, and do not mean a new algorithm can actually obtain that gain. A miss in the final output also does not mean the RPN necessarily had no proposal; score filtering, NMS and so on may also play a part. Very small detection-classification error does not mean image classification has no problem; they are different tasks.

Detailed results: [tide_errors.json](tide_errors.json). TIDE was not added to the training environment; its dependency is installed in `/tmp/cs701_autoresearch_tide`.

### Old experiments not to repeat

The table below is the old r32/head1024 36-epoch screen and cannot be directly subtracted from the final AP of the current r80/head1536.

| Method | val mAP | test mAP | Interpretation |
|---|---:|---:|---|
|AA baseline|38.429|40.009|old screening baseline|
|context3|39.446|40.841|single-factor gain|
|classification-only context3|39.816|41.329|combination with current BP7 not validated|
|spatial64 (BP7)|40.017|41.366|adopted|
|weak-class oversample2|36.724|37.732|does not support retrying the same config|
|partial negatives|38.524|40.122|close to break-even|

Source: `BP_queue/followup_20261028/comparison.json` and other records. Raising the RPN to 2000/3000 proposals on AA changed val by only about 0.01pp and slightly lowered test; SC1 box voting and removing the per-image cap of 100 boxes also gave no overall gain. In an old same-run comparison, EMA0.9998 had val mAP 0.221pp below last and test 0.024pp below. The IoU quality branch, D4/copy-paste and zoom have already been explored and should not be called brand-new directions. See the READMEs of the neighbouring experiments.

## Candidate experiments, ranked

### A: BP7 + context on the classification branch only (first training candidate)

Keep the original backbone/learning rate/schedule and add only `--roi-context 3 --roi-context-mode classification`. Classification uses a larger region; regression still uses the local region. The reason is that the old BP8 has positive evidence, and that bridge/playground are easily confused, so background semantics may be useful. The motivation is close to that of the [TSD/Sibling Head paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Song_Revisiting_the_Sibling_Head_in_Object_Detector_CVPR_2020_paper.html), which decouples classification and localization, but this implementation is not TSD and its gains cannot be assumed.

The current implementation adds one 256→64 1×1 convolution per stage, 49,344 parameters in total, for an expected 39,916,704 trained parameters and 125,126,304 total, all within budget; they must still be measured before actually starting. Only switching the mode of an existing context adds 0 new parameters. Running the RoI head an extra time adds computation and cannot be called zero cost. First test whether it overfits the background and whether regression benefits; the gain is not guaranteed to add to that of spatial64.

### B: Diagnosis-driven NWD ranking assignment (second training candidate)

The current implementation is `max(IoU,NWD)` plus a fixed Matcher, not yet the ranking assignment of [NWD-RKA](https://arxiv.org/abs/2206.13996). First count, on the latest BP7, positive anchors/positive RoIs per GT, proposal coverage excluding injected GT, and target drop-off across the Cascade stages, and then decide whether to change only the RPN matching or the Cascade thresholds.

The minimal comparison replaces one assigner only and keeps everything else fixed; refer to the per-GT adaptive assignment of NWD-RKA or [ATSS](https://arxiv.org/abs/1912.02424), and do not change sampling, loss and neck at the same time. Training-time matching adds no inference parameters. The risks are positive-sample conflicts among dense small objects, and background false positives. The low bridge/proposal coverage of the old SC1 is directional evidence, not a direct measurement of the latest BP7.

### C: Cascade localization loss (worth studying given the TIDE results)

With the assigner fixed, compare the existing encoded-box Smooth-L1 against adding a decoded-box GIoU. Check AP75/AP90, per-class localization error and small-box stability first; do not add a quality branch at the same time at first. See the [GIoU paper](https://arxiv.org/abs/1902.09630). This loss adds no inference parameters, but it may not be stable on extremely small boxes.

Implementation note: the existing `--giou` is explicitly unsupported in Cascade mode, so the flag cannot simply be turned on; it must be implemented per Cascade stage, with coordinates, positives, empty samples and normalization verified. It cannot be claimed to be ready to run.

### D: Training-strategy alternatives, keeping peak LR

1. **Align schedules by update steps.** local130=61,100 steps; local160=75,200; full130=76,310; full160=93,920. Local 160 and full 130 differ by only about 1.5%, so a longer local schedule being beneficial does not directly prove that full data also needs 160. Next time compare cosine/low-LR tails at a fixed step budget, rather than blindly going to 200 epochs. This finding does not change the full-data selection queue already explained to the user and started.
2. **No weight decay on normalization/bias.** The code already has `--wd-hygiene`; the current default still applies 0.05 decay to all trained parameters. It can serve as a single-factor comparison that changes neither the architecture nor the peak LR, but there is no positive local result supporting it yet, so its priority is below A/B. Do not attribute a classification drop directly to weight decay.
3. **Small-range post-processing calibration.** On the same forward outputs of the latest local BP7, fix a small set of classification-fusion λ / localization-stage settings and verify that F1 is kept without losing mAP. Simple classification-probability × detection-score, NMS and box voting have already given weak gains; large-grid trial scoring on the official site is rejected.

Generic teacher distillation and cross-layer LoRA rank reallocation are longer-term directions with no local evidence yet; teacher budget, the parameter union and cost need separate checks. EMA/SWA waits for rule confirmation; prediction ensembles are not done.

## Execution order and stop conditions

1. The current 160-epoch → authorized full train → submission pipeline keeps running.
2. This round of CPU research is complete, occupies no training GPU, and uploads nothing.
3. If the full-data result still justifies further investment, A has priority in the next group; try one variable only, with the local split, seed and full schedule fixed. Do not claim generalization gains on the local 940/940 that the full model has already seen.
4. Small gains need paired-seed rechecks. Submission quota goes to candidates with local evidence; Phase 2 keeps only settled plans. Do not treat a small advantage at some early epoch as the final gain.

To reproduce the CPU audit:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/tmp/cs701_autoresearch_tide MPLCONFIGDIR=/tmp/cs701_autoresearch_mpl \
TIDE_PATH=/tmp/cs701_autoresearch_tide_cache \
~/miniconda3/envs/cs701bench/bin/python tools/research_bp7_errors.py
```
