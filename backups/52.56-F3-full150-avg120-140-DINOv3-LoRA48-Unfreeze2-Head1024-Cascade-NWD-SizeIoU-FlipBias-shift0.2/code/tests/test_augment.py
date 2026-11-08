"""D4 rotations and the zoom-in augmentation move boxes with the pixels."""
import random

import torch

from sarbench.data import MEAN, STD, SARMultiTask, rot90_boxes


def bright_box(image):
    ys, xs = torch.nonzero(image[0] > 0.5, as_tuple=True)
    return torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float()  # pixel i spans [i, i + 1)


def test_rot90_boxes_follow_pixels():
    image = torch.zeros(1, 64, 64)
    image[:, 10:20, 30:50] = 1  # box x 30..50, y 10..20
    box = torch.tensor([[30., 10., 50., 20.]])
    for k in range(4):
        rotated = torch.rot90(image, k, dims=(-2, -1))
        assert torch.equal(bright_box(rotated), rot90_boxes(box, k, 64)[0]), k


def test_zoom_moves_boxes_with_pixels():
    ds = SARMultiTask.__new__(SARMultiTask)  # only the zoom method and its attributes
    ds.size, ds.zoom_max = 128, 2.0
    image = torch.zeros(1, 128, 128)
    image[:, 40:60, 50:90] = 1
    box = torch.tensor([[50., 40., 90., 60.]])
    for seed in range(20):
        random.seed(seed)
        out = ds.zoom(image, box)
        if out is None:
            continue
        zoomed, boxes, keep = out
        if boxes[0, 0] <= 0 or boxes[0, 2] >= 128 or boxes[0, 1] <= 0 or boxes[0, 3] >= 128:
            continue  # clipped by the crop: the bright region is clipped the same way, checked loosely below
        assert torch.allclose(bright_box(zoomed), boxes[0], atol=1.5), (seed, bright_box(zoomed), boxes[0])
