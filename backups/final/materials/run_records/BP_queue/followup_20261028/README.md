# Bridge / playground follow-up screens

Prepared on 2026-10-28, after the BP1/BP2/BP4 results. These are local ablations,
not leaderboard submissions. The running BP5 → BP3 queue remains unchanged.

## Order and comparisons

1. Wait for the existing queue process to exit successfully and for both BP5 and
   BP3 to have final evaluation files.
2. Diagnose the saved `AA_a4a2` model on a fixed training-only sample of at most
   16 images per class. Record positive/negative RoI counts, per-GT coverage
   with and without injected GT boxes, loss contributions and local logit-gradient
   proxies. There is no optimizer, backward pass or checkpoint modification.
3. Run the following screens sequentially, with a short training/checkpoint-load
   smoke test before each full run:

| Run | Change relative to AA | Main comparison |
| --- | --- | --- |
| `BP7_spatial64` | `--spatial-prior 64` | AA: does the old S5 spatial branch still help with A4 + flip-bias? |
| `BP8_ctx3_cls_only` | `--roi-context 3 --roi-context-mode classification` | BP1: should enlarged-context features feed classification only? |
| `BP9_ctx3_clsw3` | `--roi-context 3 --cls-weights 3:3,7:3` | BP1/BP2: do context and class weighting combine? |

All screens use the existing local train/val/test split (7,512/940/940), seed 0,
36 epochs, LoRA r32/alpha64, head width 1024, batch 16, the AA NWD/anchors/A4/
flip-bias recipe and gradient checkpointing. There is no new negative-loss rule,
pseudo-label generation, external data, or full-data training in this queue.

`classification` mode reuses the existing head weights: classification sees
local + enlarged RoI features; box regression sees the local RoI alone. It adds
head computation but no parameters. Existing checkpoints default to `shared`.

## Resources and records

- GPU0 only, one process using the GPU at a time within this queue, after its
  predecessor exits. The backbone still processes each image once.
- Nice level 10; CPU affinity limited to four CPUs; four data workers and
  library threads; at most two compiler workers. Diagnostic uses two threads
  and zero data workers.
- Source is copied to `cs701_runs/BP_queue/followup_20261028/code/`, with a SHA256
  manifest. Each training run receives another copy of this fixed source.
- Runtime state, commands, diagnostic JSON and comparison JSON live under
  `cs701_runs/BP_queue/followup_20261028/`; individual logs and checkpoints live
  under `cs701_runs/BP7_spatial64/`, `BP8_ctx3_cls_only/`, `BP9_ctx3_clsw3/`.
- A failed prerequisite, diagnostic, smoke test, training or evaluation stops the
  queue and records the error. Existing experiment directories are never reused.

## Interpretation

Compare final post-processing with the same settings (including shift 0 for the
local flip-bias models). Evaluate both local holdouts, all nine class APs and
image macro-F1. Image-class multiplication is additionally evaluated from saved
raw predictions. `comparison.json` uses the existing local reference F1/mAP:
val 91.50/22.41 and test 91.61/23.31; these are not official leaderboard references.

One seed screens ideas; it does not establish repeatability. Inspect the sampling
diagnostic before changing loss normalization. Do not interpret appended-GT
coverage as proposal recall, or a logit-gradient proxy as backbone gradient
conflict. Recheck parameter budgets and validation evidence before combining
changes or starting a scaled/full-data run.
