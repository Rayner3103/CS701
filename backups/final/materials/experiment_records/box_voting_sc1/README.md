# SC1 single-inference box-voting experiment

**English** · [中文](README.zh-CN.md)

Date: 2026-10-30 (SGT). Conclusion: none of the four preset configurations raised local val mAP, and the original post-processing stays the default. An SC2 candidate was also generated for an official val trial, see the end of this document; it has not been submitted or received an official score.

## Method and data

- Uses `~/cs701_runs/SC1_r80h1536/raw_val.pt`: 940 local holdout images. SC1 was trained only on the 7,512 images of local train; these splits all come from the official labelled training set.
- SC2 has been trained on all 9,392 labelled training images, so these local holdouts cannot be used to assess SC2's generalization. This experiment is a proxy experiment on SC1, not an official val score of SC2.
- Directly calls the frozen post-processing in the 51.86 backup: Soft-NMS σ=0.75, objectness^0.25, three-stage box mean, shift=-0.2, threshold 1e-5, min_keep=30, at most 100 boxes.
- For each box kept by Soft-NMS, from the same-class candidates of the same forward pass before suppression, select the boxes with IoU≥threshold and average their coordinates weighted by their scores not decayed by Soft-NMS. Finally apply `原框 + alpha × (加权框 − 原框)` (original box + alpha × (weighted box − original box)).
- Scores, classes, box counts and classification outputs stay the same. No re-inference, no added views, no image reading and no model training; all computation was on CPU.
- The candidate pool also calls the same frozen post-processing, changing NMS only to hard NMS with threshold 1 and removing the count cap, so that stage averaging, shift, clipping, scores and candidate filtering stay consistent.

IoU `{0.8, 0.9}` × alpha `{0.10, 0.25}` were fixed in advance. Selection is by val mAP, with the baseline preferred on ties; only a configuration beating the baseline would have the baseline and that single selected configuration compared on local test. None of the four beat the baseline this time, so local test was not used. Both holdouts have been used in other experiments before and are not treated as brand-new final test sets.

## Results

In the table below the unit is percent and differences are percentage points; macro-F1 is 95.6468 throughout, with 78,365 output boxes in total.

| Configuration | mAP | vs baseline | AP50 | AP75 |
|---|---:|---:|---:|---:|
| Original scheme, shift=-0.2 | **41.2098** | — | 68.0708 | 43.0989 |
| IoU 0.8, alpha 0.10 | 41.2040 | -0.0058 | 68.0672 | 43.1687 |
| IoU 0.8, alpha 0.25 | 41.1810 | -0.0288 | 68.0356 | 43.0722 |
| IoU 0.9, alpha 0.10 | 41.1993 | -0.0105 | 68.0765 | 43.1326 |
| IoU 0.9, alpha 0.25 | 41.2027 | -0.0071 | 68.0718 | 43.2559 |

The changes are very small, so this cannot support a claim that box voting is generally harmful, but there is no evidence for adopting these four configurations. The partial AP75 improvement did not turn into an improvement in the main metric mAP, and classification is unchanged, so the overall Δm will not improve either. The original scheme is kept this time, and test is not further used to pick configurations.

`results.json` keeps the full metrics, per-class/per-size AP, coordinate changes, and cache and code SHA256. Size AP is grouped by `sqrt(area)` at 512×512 input, with COCO interval endpoints inclusive; IoU and the main metric are still computed in original-image coordinates. Size metrics are diagnostic only and cannot replace the main metric for selection.

## Reproduce and check

Run from the repository root, with the existing `cs701bench` environment:

```bash
~/miniconda3/envs/cs701bench/bin/python -u tools/evaluate_box_voting.py --workers 2
~/miniconda3/envs/cs701bench/bin/python -m pytest -q tests/test_box_voting.py
```

14 tests pass, covering weighted coordinates, class and IoU isolation, empty input, zero weights, zero fusion, input not modified and parameter validity. An independent spot check of six cached images confirmed that kept boxes exist in the candidate pool and that the conversion to original-image coordinates agrees with the original function. The prediction JSON stays in this directory and is excluded by `.gitignore`, to avoid adding about 56 MB of regenerable results.

## SC2 official val candidate

Candidate file: `submissions/SC2_e130_val_shift-0.2_boxvote_iou0.8_alpha0.10.zip`. IoU=0.8, alpha=0.10 has the highest local val mAP of the four voting configurations, yet is still slightly below the baseline. This candidate is for further trial and does not mean it has been verified better than the original submission.

The generating script is `tools/make_box_voting_submission.py`. It takes the boxes in the original 51.86 submission as anchor boxes, recovers candidate boxes from the cache of the same SC2 e130, and adds only the voting displacement to the original coordinates; the original boxes already include shift=-0.2 and are not shifted again. The classification file is byte-for-byte unchanged, and the scores, classes, count and order of the 61,683 boxes are unchanged. Cache precision does not make these fields change on replay. The only difference between this generation method and the SC1 experiment is that the anchor boxes are taken directly from the archived submission instead of recomputing the cached post-processing.

Format check passed: 1,426 images, 61,683 boxes, no image without boxes, no out-of-bounds boxes, at most 100 boxes per class per image. All anchor boxes have a same-class candidate reaching the voting threshold. The `.json` and `.sha256` next to the candidate hold the input sources, parameters and file hash.

Later user feedback: the online effect was "almost no difference"; no exact score was provided, so no numerical gain is recorded on this basis.

```bash
~/miniconda3/envs/cs701bench/bin/python tools/make_box_voting_submission.py --out submissions/SC2_boxvote_reproduced.zip
~/miniconda3/envs/cs701bench/bin/python tools/check_submission.py submissions/SC2_boxvote_reproduced.zip ~/cs701_data/cs701-sar-course-data/val/images.json
```

## Follow-up check of small-box-only voting

After seeing the changes in size AP, with IoU=0.8, alpha=0.10 fixed, an extra check voted only on predicted boxes with `sqrt(area)<16` or `<32` (in 512 input coordinates). Grouping is by the predicted size before voting and needs no labels. Both are exploratory checks made after looking at the local val results.

| Method | val mAP | val AP75 |
|---|---:|---:|
| Original scheme | 41.2098 | 43.0989 |
| Only <16 | 41.2036 | 43.0390 |
| Only <32 | 41.2015 | 43.0381 |

Neither raised the main metric, so neither went on to the second holdout. A rise in small-size-group AP alone cannot imply that the overall metric will rise. Script `tools/evaluate_size_gated_voting.py`; detailed results `size_gate_results.json`.

## Next diagnosis: RPN proposal coverage

On the same SC1 local val cache, take the RPN boxes of `raw_stage[0]` and sort them by `raw_obj`. Each GT is checked independently for a proposal reaching IoU≥0.5; no classification-score threshold and no one-to-one matching are used. So this is proposal coverage, neither AP nor final detection recall.

| Target group | GT count | Coverage by top 500 proposals | Coverage by top 1000 proposals |
|---|---:|---:|---:|
| All | 3323 | 88.44% | 89.77% |
| <16 pixels | 1698 | 80.74% | 82.51% |
| bridge | 357 | 45.94% | 48.74% |
| playground | 276 | 56.88% | 66.30% |

The current RPN pre/post-NMS cap is 2000 in training and 1000 in testing (frozen `code/sarbench/model.py:202`). The latter half of the proposals still covers additional targets, which supports listing "raise the test proposal cap to 2000" as the next experiment; it cannot be concluded from this that mAP must rise, and new proposals may also add false positives. This needs re-inference and cannot be filled in from the existing 1000-proposal cache. This round changed no model and started no new GPU inference. Diagnostic data is in `proposal_coverage.json`.
