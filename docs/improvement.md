1. Look at bridge and playground images first. Run visualise.py on some ground-truth boxes next to your predictions. A 15 px "playground" is odd; check whether the labels mark something else, come from a different source or resolution, or are noisy. What you find decides whether to work on anchors, input resolution or data cleaning.
2. Spend the unused trainable budget. You train 21.1 M parameters out of 40 M, and full fine-tuning is +13.5 Δm over LoRA.
   - The box head's first layer alone (TwoMLPHead fc6, 12544×1024) is 12.8 M parameters.
   - Replacing it with a 4-conv + 1-fc head frees most of that budget.
   - Spend the room on the backbone: add LoRA to the MLP layers fc1/fc2, raise the rank, or unfreeze the last 2–3 blocks (about 7.1 M each).
3. Better classification pooling. model.py drops the CLS token and averages all 1024 tokens, which washes out images whose only signal is a few 10 px objects. Try a learned attention-pooling query, or GeM / max+mean pooling, plus label smoothing. It's a cheap change and targets the F1 gap.
4. Stronger training augmentation (still allowed under the no-TTA rule): you only use flips now.
   - 90° rotations: exact for overhead images, and boxes stay axis-aligned.
   - Scale jitter / random resized crop, which shows small objects at larger sizes.
   - Copy-paste of small objects.
5. Tighter boxes for the AP75 gap: GIoU/DIoU box loss, Cascade R-CNN-style refinement stages, Soft-NMS.
6. Free settings:
   - Lower score_thresh from 0.05 to 0.001 at eval; this is standard practice for COCO mAP.
   - Keep an EMA of the weights; it's still one model.
   - Consider anchors of about 4–6 px, since the smallest 10% of wind turbines are about 5.5 px.
7. Final submission: retrain on 100% of train, using the epoch count chosen on your local split. prepare_course_data.py holds out 20%.

To iterate faster, run ablations at 24 epochs. Seed noise is about ±0.9 Δm, so trust only differences above about 1.5.
