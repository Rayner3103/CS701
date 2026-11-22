# SC1 ablation of the total detection-box cap

**English** · [中文](README.zh-CN.md)

2026-10-30 (SGT). Conclusion: removing the per-image cap of 100 boxes did not raise local val mAP; the original cap stays, and the local test confirmation stage was not entered.

## Hypothesis and setup

The final post-processing has both `per_class=100` and `total=100`. In SC1 val, 603/940 images reach the per-image total cap, of which 345 contain several predicted classes. The question is whether wrong classes take the candidate slots of the correct class.

Only the original setting and `total=None` are compared, still keeping 100 boxes per class. Everything else is identical: the frozen post-processing of the SC2 backup, Soft-NMS σ=0.75, objectness^0.25, three-stage box mean, shift=-0.2, threshold=1e-5, min_keep=30. Classification stays λ=1. No training, no re-inference, no GPU.

The data is `~/cs701_runs/SC1_r80h1536/raw_val.pt`, 940 local holdout images, all from the official labelled training set. The script checks that the training and holdout image IDs are disjoint, and asserts image by image that the first 100 items of the extended output are exactly the same as the baseline in coordinates, class and score. Judged by val mAP; only an increase would confirm the same fixed scheme on the second holdout.

## Results

| Configuration | mAP | AP50 | AP75 | Output boxes |
|---|---:|---:|---:|---:|
| 100 per class, 100 per image | **41.2098** | 68.0708 | 43.0989 | 78,365 |
| 100 per class, no per-image limit | 41.2073 | 68.0699 | 43.1009 | 129,283 |

Removing the total cap adds about 65% boxes, and mAP falls by about 0.0025 percentage points, close to flat but with no gain. AP of all size tiers is also slightly lower. There is no reason to enlarge the submission file for these extra predictions. macro-F1 is unchanged (95.6468). No official submission is generated this time.

To reproduce:

```bash
~/miniconda3/envs/cs701bench/bin/python -u tools/evaluate_detection_cap.py --workers 2
```

`results.json` saves the full metrics, sources and code hashes. Regenerable prediction JSON is excluded by this directory's `.gitignore`. The definition of size AP follows `tools/evaluate_box_voting.py`, grouped by area at 512 input, and the main metric still uses IoU in original-image coordinates.
