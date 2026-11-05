# BP7 full-data large model: 130-epoch dual-GPU training

**English** · [中文](README.zh-CN.md)

Per the user's instruction, this run combines the SC2 large-model configuration (LoRA rank 80 / alpha 160, Cascade head 1536) with BP7's SpatialPrior(64).

- Initialized from generic DINOv3 pretrained weights, with new modules trained from scratch; neither the BP7 small model nor SC2 fine-tuned weights are loaded.
- Trains only on all 9,392 images of the official train. The local train+val+test are internal splits of the official train, checked set by set.
- Formal training is 130 epochs with a 130-epoch cosine learning-rate decay; it is not epoch 130 of the old SC2's 150-epoch plan.
- GPU0, GPU1, DDP global batch16, 8 per GPU; total CPU limited to 4 cores, 2 data workers per rank.
- 125,076,960 total parameters, 39,867,360 trainable; gradient checkpointing, compile, A4, NWD8 and flip-bias -0.3 carry over from SC2.
- Before formal training, a dual-GPU 3-step smoke test verifies loading of saved weights and the prediction format on 8 official val images. Smoke weights are not used for formal training.
- A checkpoint is saved every 10 epochs, and the final epoch 130 is saved as model.pt.
- On completion, a shift -0.2 prediction ZIP and raw outputs for the official Phase 1 val are generated automatically, saved locally only, not uploaded automatically.

Run status is in status.json, the scheduler log in driver.log, formal-training logs in stdout.log/log.txt, the code snapshot in code/, the hash manifest in source.sha256. Launch arguments are in commands.json.
