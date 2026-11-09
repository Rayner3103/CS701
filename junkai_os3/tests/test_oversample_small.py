"""--oversample-small (the rayner branch's os3, ported): images with a box of area < 32^2 px on the 512 input appear N
times in the training set only; the split builder and the local scorer used by gpuClusterRun/junkai_best_os3.sh."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sarbench.data import SARMultiTask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import local_eval  # noqa: E402
import make_vallike_split  # noqa: E402


def tiny_dataset(root, boxes_per_image, size=(64, 64)):
    """boxes_per_image: list of lists of (w, h) in original pixels (images are `size`, class 0)."""
    (root / 'classification').mkdir(parents=True)
    (root / 'detection').mkdir()
    (root / 'train' / 'images').mkdir(parents=True)
    images, annotations, rows = [], [], []
    for i, boxes in enumerate(boxes_per_image):
        name = f'train/images/{i}.png'
        Image.fromarray(np.zeros(size[::-1], dtype=np.uint8)).save(root / name)
        images.append({'id': i, 'file_name': name, 'width': size[0], 'height': size[1]})
        rows.append({'image': name, 'label': 0})
        annotations += [{'id': 100 * i + k, 'image_id': i, 'category_id': 1, 'bbox': [1, 1, w, h], 'area': w * h, 'iscrowd': 0}
                        for k, (w, h) in enumerate(boxes)]
    with open(root / 'classification' / 'train.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'label'])
        w.writeheader()
        w.writerows(rows)
    (root / 'detection' / 'instances_train.json').write_text(json.dumps(
        {'images': images, 'annotations': annotations, 'categories': [{'id': 1, 'name': 'aircraft'}]}))


def test_small_images_repeated_in_training_only(tmp_path):
    # a 64 px image is stretched x8 to 512: a 3x3 box becomes 24x24 (small, 576 < 1024), a 5x5 box 40x40 (not small)
    tiny_dataset(tmp_path, [[(3, 3)], [(5, 5)], [(5, 5), (3, 3)], [(8, 8)]])
    plain = SARMultiTask(tmp_path, 'train', train=True)
    over = SARMultiTask(tmp_path, 'train', train=True, oversample_small=3)
    assert len(plain) == 4 and over.n_small == 2 and len(over) == 4 + 2 * 2
    names = [r['image'] for r in over.rows]
    assert [names.count(f'train/images/{i}.png') for i in range(4)] == [3, 1, 3, 1]
    assert [r['image'] for r in over.rows[:4]] == [r['image'] for r in plain.rows]  # originals first, always kept
    held_out = SARMultiTask(tmp_path, 'train', train=False, oversample_small=3)
    assert len(held_out) == 4 and held_out.n_small == 0  # val/test are never oversampled


def test_oversample_small_one_is_a_noop(tmp_path):
    tiny_dataset(tmp_path, [[(3, 3)], [(9, 9)]])
    ds = SARMultiTask(tmp_path, 'train', train=True, oversample_small=1)
    assert len(ds) == 2 and ds.n_small == 0


def test_threshold_follows_stretch_to_512(tmp_path):
    # 256x128 image stretched to 512x512: a 16x16 box becomes 32x64 = 2048 (not small), 8x8 becomes 16x32 = 512 (small)
    tiny_dataset(tmp_path, [[(16, 16)], [(8, 8)]], size=(256, 128))
    ds = SARMultiTask(tmp_path, 'train', train=True, oversample_small=2)
    assert ds.n_small == 1 and [r['image'] for r in ds.rows] == ['train/images/0.png', 'train/images/1.png', 'train/images/1.png']


def test_item_still_loads_for_a_copy(tmp_path):
    tiny_dataset(tmp_path, [[(3, 3)]])
    ds = SARMultiTask(tmp_path, 'train', train=True, oversample_small=3)
    image, label, target = ds[2]
    assert image.shape == (1, 512, 512) and label == 0 and target['boxes'].shape == (1, 4)


# ---- vallike split builder ----
def test_vallike_cells_add_up():
    assert sum(t for t, _ in make_vallike_split.CELLS.values()) == make_vallike_split.N_VALLIKE == 992
    assert all(t <= a for t, a in make_vallike_split.CELLS.values())


# ---- local scorer ----
def test_delta_m_matches_the_report_formula():
    assert local_eval.delta_m(93.31, 26.13) == 0
    assert abs(local_eval.delta_m(95.27, 55.10) - 56.49) < 0.05  # the official Phase 1 entry


def perfect(gts, n):
    dets = {i: (np.array(g, dtype=float), np.ones(len(g)), np.full(len(g), 1)) for i, g in gts.items()}
    return dets


def test_map_is_100_for_exact_boxes_and_drops_for_shifted_ones():
    gts = {0: [[10, 10, 8, 8], [40, 40, 6, 6]], 1: [[5, 5, 20, 12]]}
    gt_arrays = {i: {1: np.array(g, dtype=float)} for i, g in gts.items()}
    ids = [0, 1]
    m = local_eval.coco_map({i: local_eval.match_image(d, gt_arrays[i], [1]) for i, d in perfect(gts, 2).items()}, ids, [1])
    assert abs(m[0] - 1.0) < 1e-9 and abs(m[1] - 1.0) < 1e-9
    shifted = {i: (d[0] + np.array([2, 0, 0, 0]), d[1], d[2]) for i, d in perfect(gts, 2).items()}
    m2 = local_eval.coco_map({i: local_eval.match_image(d, gt_arrays[i], [1]) for i, d in shifted.items()}, ids, [1])
    assert m2[0] < 0.6 and m2[1] > 0.9  # AP50 survives a 2 px shift, strict IoU does not


def test_integer_snap_applies_to_aircraft_and_256px_ship_only():
    dets = {0: (np.array([[10.4, 20.6, 5.2, 5.0], [10.4, 20.6, 5.2, 5.0]]), np.ones(2), np.array([1, 7])),   # aircraft, playground
              1: (np.array([[10.4, 20.6, 5.2, 5.0]]), np.ones(1), np.array([8]))}                              # ship
    sizes = {0: (512, 512), 1: (256, 256)}
    out = local_eval.snap(dets, sizes)
    assert np.allclose(out[0][0][0], [10, 21, 6, 5])        # aircraft: corners rounded (x 10.4->10, 15.6->16; y 20.6->21, 25.6->26)
    assert np.allclose(out[0][0][1], dets[0][0][1])          # playground untouched
    assert np.allclose(out[1][0][0], [10, 21, 6, 5])        # ship on a 256 px image
    big_ship = local_eval.snap({1: dets[1]}, {1: (512, 512)})
    assert np.allclose(big_ship[1][0], dets[1][0])           # ship on a 512 px image: not an integer cell


def test_wind_lattice_snaps_only_when_the_offset_is_consistent():
    off = 0.3
    xs = np.array([10, 40, 80, 120]) + off
    good = np.stack([xs + 0.02, xs * 0 + 7 + off + 0.03, np.full(4, 6.0), np.full(4, 6.0)], axis=1)
    dets = {0: (good, np.ones(4), np.full(4, 9))}
    out = local_eval.lattice_snap(dets)
    frac = out[0][0][:, 0] % 1
    assert np.ptp(frac) < 1e-6 and abs(frac[0] - (off + 0.02)) < 1e-6 and np.allclose(out[0][0][:, 2], 6)  # one shared lattice
    rng = np.random.default_rng(0)
    scattered = np.stack([rng.uniform(0, 400, 6), rng.uniform(0, 400, 6), np.full(6, 6.3), np.full(6, 5.7)], axis=1)
    dets = {0: (scattered, np.ones(6), np.full(6, 9))}
    assert np.allclose(local_eval.lattice_snap(dets)[0][0], scattered)  # R < 0.5: unchanged


def test_macro_f1():
    labels, pred = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    # class 0: tp 1 fp 0 fn 1 -> .667; class 1: tp 2 fp 1 fn 0 -> .8; the other 7 classes score 0 -> mean over 9
    assert abs(local_eval.macro_f1(labels, pred) - 100 * (2 / 3 + 0.8) / 9) < 1e-6
    assert torch.tensor(1).item() == 1
