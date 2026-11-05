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

Run directory: `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full130_r80h1536_spatial64`.
Launch driver: `tools/run_bp7_full130.py`.

## Final results (2026-10-29, Phase 1)

Training and prediction finished at 20:57. Formal training took 6.1304 hours on 2 GPUs; the peak training GPU memory recorded by rank0 was 11.77 GiB. The online scores below were provided by the user through screenshots or JSON; the table keeps two decimals, and the original precision is in the corresponding `leaderboard_score.json` in the run directory.

| Checkpoint | BP7 Δm | BP7 mAP | SC2 Δm | SC2 mAP | BP7 Δm difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| e40 | 32.89 | 42.68 | 30.70 | 41.45 | +2.19 |
| e80 | 48.40 | 50.75 | 45.83 | 49.37 | +2.57 |
| e110 | 51.60 | 52.45 | not scored | not scored | — |
| e130 | 51.76 | 52.53 | 51.86 | 52.66 | −0.10 |

BP7 e130: Macro-F1 95.65%, Accuracy 95.93%, AP50 73.32%. Relative to e110, Δm rose by about 0.16 and mAP by about 0.08, while AP50 fell by about 0.16.

Conclusion of this run: BP7's early same-epoch advantage did not turn into a final score above the old best; the best known so far is still SC2 e130 (51.86). The final gap of 0.10 is not enough to prove that either scheme is stably better. The two use 130-epoch and 150-epoch cosine plans respectively, with different data-worker counts, and only one run each; the difference cannot all be attributed to the spatial prior. e40/e80/e130 come from the same training trajectory and are not independent seed rechecks.

Final submission: `submissions/BP7_full130_r80h1536_spatial64_e130_val_shift-0.2.zip`, 1,426 images, shift −0.2. Format and SHA256 are verified; all uploads are done by the user.

Final weights: `model.pt` in the run directory; online result summary: `leaderboard_summary.json` in the run directory; all checkpoints and the code snapshot are kept.
