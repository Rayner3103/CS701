"""The course data in TA Zichen's folder layout (see README): every image has one class label and >= 1 box.
Loads images and boxes at a fixed input size and maps predicted boxes back to original pixels."""
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.ops import box_convert
from torchvision.transforms import v2

# Mean and std of all pixels of the 9,392 train images, read as gray, resized to
# 512 x 512 and scaled to [0, 1] (computed once).
MEAN, STD = 0.2526, 0.2133


def rot90_boxes(boxes, k, size):
    """XYXY boxes on a size x size image rotated like torch.rot90(image, k, dims=(-2, -1))."""
    for _ in range(k % 4):
        x1, y1, x2, y2 = boxes.unbind(-1)
        boxes = torch.stack([y1, size - x2, y2, size - x1], dim=-1)
    return boxes


class SARMultiTask(Dataset):
    """The images of classification/{split}.csv with their boxes from detection/instances_{split}.json.
    An item is (image [1, size, size], label 0..8, target), target = {'boxes': XYXY on the resized
    image, 'labels': 1..9, 'image_id': COCO id, 'orig_size': (h, w)}.
    split may join several splits with '+', e.g. 'train+val+test' (the final model trains on all labelled data).
    aug (training only): 'flip' (reference: horizontal and vertical flips), 'd4' (flips and 90-degree rotations:
    the 8 symmetries of the square, exact for boxes). zoom_p > 0 adds a random zoom-in (crop of 1/s of the image,
    s ~ U(1, zoom_max), resized back to size; boxes cut by less than half are kept, clipped)."""

    def __init__(self, root, split, train=False, size=512, limit=None, aug='flip', zoom_p=0.0, zoom_max=1.5):
        self.root = Path(root)
        self.rows, self.coco_images, self.annotations = [], {}, defaultdict(list)
        for part in split.split('+'):
            with open(self.root / 'classification' / f'{part}.csv') as f:
                self.rows += list(csv.DictReader(f))
            coco = json.loads((self.root / 'detection' / f'instances_{part}.json').read_text())
            self.coco_images.update({img['file_name']: img for img in coco['images']})  # key = CSV 'image'
            for ann in coco['annotations']:
                self.annotations[ann['image_id']].append(ann)
        self.rows = self.rows[:limit]
        self.size, self.train, self.aug = size, train, aug if train else 'none'
        self.zoom_p, self.zoom_max = (zoom_p, zoom_max) if train else (0.0, 1.0)
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train and aug in ('flip', 'd4') else []
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat('XYXY'),  # COCO [x, y, w, h] -> [x1, y1, x2, y2]
            v2.Resize((size, size)),  # stretches non-square images; boxes are scaled with the pixels
            *flips,  # overhead imagery has no canonical "up"
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([MEAN], [STD]),
        ])

    def __len__(self):
        return len(self.rows)

    def zoom(self, image, boxes):
        """Random zoom-in: crop a (size / s)^2 window and resize it back to size x size. Returns the new image and
        boxes, and the indices of the boxes kept; or None if no box would survive."""
        s = random.uniform(1.0, self.zoom_max)
        crop = self.size / s
        x0, y0 = random.uniform(0, self.size - crop), random.uniform(0, self.size - crop)
        new = (boxes - boxes.new_tensor([x0, y0, x0, y0])) * s
        clipped = new.clamp(0, self.size)
        area = (new[:, 2] - new[:, 0]) * (new[:, 3] - new[:, 1])
        area_c = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
        keep = (area_c >= 0.5 * area) & (clipped[:, 2] - clipped[:, 0] >= 1) & (clipped[:, 3] - clipped[:, 1] >= 1)
        if not keep.any():
            return None
        # crop + resize as one bilinear resample of the window
        theta = image.new_tensor([[1 / s, 0, (2 * x0 + crop) / self.size - 1], [0, 1 / s, (2 * y0 + crop) / self.size - 1]])
        grid = F.affine_grid(theta[None], (1, 1, self.size, self.size), align_corners=False)
        out = F.grid_sample(image[None], grid, mode='bilinear', padding_mode='border', align_corners=False)[0]
        return out, clipped[keep], keep

    def __getitem__(self, i):
        row = self.rows[i]
        info = self.coco_images[row['image']]
        anns = self.annotations[info['id']]
        orig_size = (info['height'], info['width'])
        boxes = tv_tensors.BoundingBoxes([a['bbox'] for a in anns], format='XYWH', canvas_size=orig_size,
                                         dtype=torch.float32)  # integer boxes would be truncated by Resize
        image = Image.open(self.root / row['image']).convert('L')  # PIL: torchvision.io cannot read .bmp
        image, boxes = self.transform(image, boxes)
        labels = torch.tensor([a['category_id'] for a in anns])
        boxes = boxes.as_subclass(torch.Tensor)
        if self.aug == 'd4':
            k = random.randrange(4)
            if k:
                image = torch.rot90(image, k, dims=(-2, -1))
                boxes = rot90_boxes(boxes, k, self.size)
        if self.zoom_p and random.random() < self.zoom_p:
            zoomed = self.zoom(image, boxes)
            if zoomed is not None:
                image, boxes, keep = zoomed
                labels = labels[keep]
        target = {'boxes': boxes, 'labels': labels, 'image_id': info['id'], 'orig_size': orig_size}
        return image, int(row['label']), target


class SARImages(Dataset):
    """Unlabelled images listed in an images.json (Codabench val/test), preprocessed exactly as the eval path of
    SARMultiTask. An item is (image [1, size, size], {'image_id', 'orig_size'})."""

    def __init__(self, root, images, size=512):
        self.root, self.images = Path(root), images
        self.transform = v2.Compose([v2.ToImage(), v2.Resize((size, size)), v2.ToDtype(torch.float32, scale=True),
                                     v2.Normalize([MEAN], [STD])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        image = self.transform(Image.open(self.root / info['file_name']).convert('L'))
        return image, {'image_id': info['id'], 'orig_size': (info['height'], info['width'])}


def collate(batch):
    """Stack images and labels; targets stay a list because images have different numbers of boxes."""
    images, labels, targets = zip(*batch)
    return torch.stack(images), torch.tensor(labels), list(targets)


def to_original_xywh(boxes_xyxy, orig_size, size=512):
    """Undo the resize: XYXY boxes on the size x size input -> COCO [x, y, w, h] in original pixels."""
    h, w = orig_size
    return box_convert(boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h]) / size, 'xyxy', 'xywh')
