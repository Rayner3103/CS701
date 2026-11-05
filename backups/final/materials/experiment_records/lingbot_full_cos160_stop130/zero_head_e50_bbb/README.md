# Zero-training assembled submission bbb.zip

**English** · [中文](README.zh-CN.md)

The user asked to swap in LingBot-B with no training at all and generate a submission. To compare with aaa.zip, the classification head, neck, RPN and Cascade of the same BP7 full e50 are reused as fixed, 88 tensors and 28032480 parameters in total; no DINOv3 backbone or adapter tensors are copied.

LingBot uses the official Base pretraining, pinned and SHA256-verified. New LoRA up is zero, and all original backbone tensors are checked item by item to match the official weights. After assembly no optimizer was created and no training update run; at inference all parameters are frozen, eval/no_grad. Input, official val 1426, batch2 and post-processing match aaa, with explicit shift−0.2.

submissions/bbb.zip and bbb.sha256 have been generated and pass the format, image coverage, finite values, box range, 100 boxes per image, ZIP integrity and 25MB limit validation. Not uploaded. Source and model validation are in assembly.json, the generating command in launch.json, submission validation in validation.json.

The first line of model/log.txt merely carries over the fields needed to rebuild the architecture and the training recipe of the head donor; it does not mean the assembled model was trained for those epochs again. New training steps after assembly are 0; the donor head was already trained for 50 epochs on the official train 9392. prediction_comparison.json compares only unlabelled predictions and cannot be treated as an accuracy or generalization metric.
