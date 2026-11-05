# SC2 e130: Phase 1 submission with 2000 RPN proposals

**English** · [中文](README.zh-CN.md)

2026-10-30 (SGT). Local file generated and checked; not yet uploaded and no official score received.

Submission: `submissions/SC2_e130_val_shift-0.2_rpn2000.zip`.

The model comes from `weights/model.pt` of the 51.86 backup, loading the same frozen DINOv3 pretrained backbone and trainable weights. All inference source uses that backup's `code/`; other training changes in the current working tree are not mixed in.

The only model config change: the RPN testing pre-NMS and post-NMS caps are both raised from 1000 to 2000. The training cap stays 2000 and the RPN NMS threshold stays 0.7. Input 512×512, bf16, classification fusion λ1, Soft-NMS σ0.75, objectness^0.25, three-stage box mean, shift=-0.2, score threshold 1e-5, min_keep30, and a final maximum of 100 boxes per class/per image all keep the original settings, with no box voting.

One forward pass produces classification and detection together. Classification still uses the detection-assisted fusion from the same 2000-proposal output and does not copy classification results from the old submission; a final check shows the classification CSV is exactly byte-for-byte identical to the original submission.

## Running and checks

- GPU1, batch2, 2 workers, 2 CPU threads, nice10; PyTorch GPU memory allocation capped at 12% of device total. Ongoing training was not stopped or modified.
- All 1426 images completed, 713 batch forward passes in total, each image input only once; the cache confirms every image has 2000 RPN proposals entering Cascade.
- Parameters remain 124,866,272 total, 39,656,672 trainable.
- End-to-end model prediction is about 53.2ms/image; the record including loading and file writing is about 78.23 seconds. Peak GPU memory is about 0.96GiB. The GPU was shared with other processes, so this is not an exclusive performance test.
- Compared with the original default batch16, batch2 is used to control resources; the bf16 batch-size change may bring slight numerical rounding differences.
- Submission check passed: 1426 images, 61,896 boxes, no image without boxes, no out-of-bounds boxes, at most 100 boxes per class per image.
- The original submission has 61,683 boxes and this one 61,896; 0 images change classification label.
- ZIP about 9.56MB, SHA256: `06d9b37e934c43632aaf0067d8e517b5e911e248fd63d15cc1e13b238513defa`.

`manifest.json` saves the model/source/input/output hashes, full config, and actual forward and proposal counts; `status.json` records the completion state. Raw outputs are kept in `submissions/raw/SC2_e130_val_rpn2000_raw.pt` for later CPU analysis.

## Reproduce

Run from the repository root. First choose a GPU with headroom; output paths must not exist:

```bash
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/cs701bench/bin/python -u tools/make_rpn2000_submission.py \
  --out submissions/SC2_e130_rpn2000_reproduced.zip \
  --raw-out submissions/raw/SC2_e130_rpn2000_reproduced.pt \
  --record-dir experiments/sc2_rpn2000_reproduced
~/miniconda3/envs/cs701bench/bin/python tools/check_submission.py \
  submissions/SC2_e130_rpn2000_reproduced.zip \
  ~/cs701_data/cs701-sar-course-data/val/images.json
```
