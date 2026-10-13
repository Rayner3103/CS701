# Browsing predictions: visualize.py

Back to the [README](../README.md).

`visualize.py` loads a trained run's `best_model.pt`, predicts on a random sample of a labelled local split and
opens a viewer: ground truth on the left, predictions on the right.

## Commands

```bash
# one run
python visualize.py --run runs/single_seed0_ep100/terramind_pretrained_moelora

# a folder of runs: lists them with their best val macro-F1 and mAP, then asks for a number
python visualize.py --run runs/full_seed0

# only images of some classes
python visualize.py --run runs/full_seed0 --class bridge playground --n 100 --seed 1

# no window (e.g. over SSH): one PNG per sampled image, then exit
python visualize.py --run runs/full_seed0/vit_pretrained_lora_sp --save vis/
```

Run it from the repository root, in the `cs701bench` environment. It uses the GPU if there is one, else the CPU.

## Options

| option | default | meaning |
|---|---|---|
| `--run` | (required) | a run folder of `train.py`, or a folder of run folders (you then pick one) |
| `--weights` | `best_model.pt` | the best-val-Δm model; `model.pt` for the last epoch (older runs have only `model.pt`) |
| `--data` | `../dataset/SARFact-Course-20K` | the prepared course data, as for `train.py` |
| `--split` | `val` | `train`, `val` or `test`: a local split with labels (`prepare_course_data.py`) |
| `--class` | all | sample only images whose true class is one of these names; an unknown name lists the valid ones |
| `--n` | 50 | sample size |
| `--seed` | 0 | which random sample; the same seed gives the same images |
| `--score-thresh` | 0.3 | starting score threshold for the predictions shown (change it in the viewer) |
| `--iou` | 0.5 | IoU a prediction needs with a true box of its class to count as matched |
| `--batch-size` | 8 | inference batch size |
| `--save` | off | write PNGs to this folder instead of opening a window |

Class names: aircraft, airport, bridge, car, harbor, oil_tank, playground, ship, wind_turbine.

## Keys

| key | action |
|---|---|
| ← / → | previous / next image |
| ↑ / ↓ | raise / lower the score threshold by 0.05 (keeps the zoom) |
| `e` | only images with an error (wrong class, a missed box or a false positive) on / off |
| `v` | overlay on / off: true boxes (thin) and predictions (thick) in one panel |
| `l` | box labels on / off |
| `s` | save the current view as a PNG in the current folder |
| `q` | quit |

Zoom with the toolbar's magnifier: both panels zoom together, which helps with objects of 6-15 px. The home button
resets the zoom.

## Reading the view

- **Title:** run, position in the sample, file and its size; the true and predicted class (red when wrong) and the
  top-3 class probabilities.
- **Colours:** one per class (legend at the bottom).
- **Solid box:** matched, i.e. a prediction and a true box of the same class with IoU >= `--iou`. Matching is greedy
  in descending score order, as in COCO.
- **Dashed box:** in the left panel, a true box that no prediction above the threshold matched (missed); in the
  right panel, a prediction that matched no true box (false positive).
- **Panel titles:** the number of true boxes and missed ones, and the number of predictions above the threshold and
  false positives. They change with the threshold.
- **Boxes are drawn on the original image at full resolution.** The model sees a 512 × 512 resize, which stretches
  non-square images.

## Tips

- Low-AP classes (bridge, playground): `--class bridge playground`, then `e` to page through the errors only.
- Dashed predictions that look like real objects point to missing labels rather than model errors.
- Comparing two runs: run both with the same `--seed` and `--class` to see the same images.
