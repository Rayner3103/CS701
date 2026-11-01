"""Small-object oversampling and copy-paste (Kisantal et al. 2019, arXiv 1902.07296) on the YOLO data of
prepare_yolo_data.py, written offline to a new folder; the source folder is not touched.

  1. Oversampling: every image with at least one small box appears --oversample times in all (the original plus
     --oversample - 1 extra copies; extra copies are symlinks to the same image file, so they cost no disk).
  2. Copy-paste: each extra copy is, with probability --paste-prob, a copy-pasted version: small boxes of the same
     image are copied to new places (so the image class never changes). 'multi' (default): 1..--max-objs small boxes,
     each pasted 1..--max-copies times; 'all': every small box once. Each pasted copy is scaled by +-20 % and rotated by
     +-15 degrees, must not touch any other box, sits >= 5 px from the border, and is pasted hard-edged (no blur).
     Only boxes that are not clipped by the border and touch no other box are copied (the paper's "non-occluded").
  The originals are always kept ("original + aug": originals to augmented about 2:1 with --oversample 2 and
  --paste-prob 0.5). Small = box area below --small-px squared in an image resized to --ref-size (the size the
  detector sees: 256 px images are enlarged), so a 256 px image's box counts at twice its stored pixel size.
Splits not listed in --splits are linked, not copied; keep val on the original images to score a run.
Writes DST/data_aug.yaml (train on the augmented train, val on the original val) and DST/data_full_aug.yaml (train on
train + val + test, the final submission model).
Example: python augment_small_objects.py --oversample 2 --paste-prob 0.5
         python augment_small_objects.py --oversample 3 --paste-prob 0   (oversampling only)
         python augment_small_objects.py --paste-mode all --dst ../dataset/SARFact-YOLO-aug-all"""
import argparse
import json
import math
import random
import shutil
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

SPLITS = ('train', 'val', 'test')
SCALE = 0.2       # pasted copy size: x [0.8, 1.2]
ANGLE = 15.0      # degrees
MARGIN = 5        # px from the image border
TRIES = 50        # placements tried per pasted copy


def read_labels(path):
    """'class cx cy w h' lines (normalised) -> int array [N] and float array [N, 4]; empty when there is no file."""
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    arr = np.array(rows, float)
    return arr[:, 0].astype(int), arr[:, 1:5]


def small_mask(boxes, small_px, ref_size):
    """Boxes (normalised cx cy w h) whose area is below small_px^2 when the image is ref_size px wide and high."""
    return boxes[:, 2] * boxes[:, 3] * ref_size ** 2 < small_px ** 2


def to_xyxy(boxes, width, height):
    """Normalised cx cy w h -> x1 y1 x2 y2 in pixels."""
    cx, cy, w, h = (boxes[:, i] for i in range(4))
    return np.stack([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height], 1)


def free_of_neighbours(xyxy, tol=0.5):
    """[N] bool: the box shares no area (more than tol px^2) with any other box; for boxes this stands in for the
    paper's 'non-occluded' test (which uses instance masks)."""
    n = len(xyxy)
    ok = np.ones(n, bool)
    for i in range(n):
        for j in range(n):
            if i != j and overlap_area(xyxy[i], xyxy[j]) > tol:
                ok[i] = False
                break
    return ok


def overlap_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def transformed_patch(image, box, rng):
    """Crop box (x1 y1 x2 y2 px) from the image, scale it by 1 +- SCALE, rotate it by +- ANGLE degrees. Returns the
    patch (cropped to the rotated rectangle's bounding box) and its mask (255 inside the rotated rectangle), or None
    when the crop is empty."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    crop = image[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = 1 + rng.uniform(-SCALE, SCALE)
    angle = rng.uniform(-ANGLE, ANGLE)
    rad = math.radians(abs(angle))
    out_w = max(1, int(math.ceil(scale * (w * math.cos(rad) + h * math.sin(rad)))))
    out_h = max(1, int(math.ceil(scale * (w * math.sin(rad) + h * math.cos(rad)))))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += out_w / 2 - w / 2
    m[1, 2] += out_h / 2 - h / 2
    patch = cv2.warpAffine(crop, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (out_w, out_h), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))  # tight box of the pasted pixels
    return patch[sl], mask[sl]


def place(size, occupied, width, height, rng):
    """A top-left (x, y) for a size = (w, h) patch so that it is MARGIN px from the border and overlaps no box in
    occupied (x1 y1 x2 y2 px); None after TRIES tries."""
    pw, ph = size
    if pw + 2 * MARGIN > width or ph + 2 * MARGIN > height:
        return None
    for _ in range(TRIES):
        x = rng.randint(MARGIN, width - MARGIN - pw)
        y = rng.randint(MARGIN, height - MARGIN - ph)
        cand = (x, y, x + pw, y + ph)
        if all(overlap_area(cand, o) == 0 for o in occupied):
            return cand
    return None


def paste_small(image, classes, boxes, args, rng):
    """Copy-paste small boxes of one image. Returns the new image, the extended classes and boxes (normalised), and the
    number of pasted objects and of failed placements."""
    height, width = image.shape[:2]
    xyxy = to_xyxy(boxes, width, height)
    eligible = np.nonzero(small_mask(boxes, args.small_px, args.ref_size) & free_of_neighbours(xyxy)
                          & (xyxy[:, 0] > 0) & (xyxy[:, 1] > 0) & (xyxy[:, 2] < width) & (xyxy[:, 3] < height))[0].tolist()
    if not eligible:
        return image, classes, boxes, 0, 0
    if args.paste_mode == 'all':
        jobs = [i for i in eligible]
    else:
        picked = rng.sample(eligible, rng.randint(1, min(args.max_objs, len(eligible))))
        jobs = [i for i in picked for _ in range(rng.randint(1, args.max_copies))]
    out = image.copy()
    occupied = [tuple(b) for b in xyxy]
    new_classes, new_boxes, failed = [], [], 0
    for i in jobs:
        piece = transformed_patch(image, xyxy[i], rng)  # always cut from the original, never from an earlier paste
        spot = place(piece[0].shape[1::-1], occupied, width, height, rng) if piece else None
        if spot is None:
            failed += 1
            continue
        patch, mask = piece
        x1, y1, x2, y2 = spot
        region = out[y1:y2, x1:x2]
        region[mask > 0] = patch[mask > 0]
        occupied.append(spot)
        new_classes.append(classes[i])
        new_boxes.append([(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height])
    if new_boxes:
        classes = np.concatenate([classes, new_classes])
        boxes = np.concatenate([boxes, np.array(new_boxes)])
    return out, classes, boxes, len(new_boxes), failed


def write_labels(path, classes, boxes):
    path.write_text(''.join(f'{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n' for c, b in zip(classes, boxes)))


def link(link_path, target):
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target)


def augment_split(src, dst, args):
    """Write dst/images and dst/labels from src/images and src/labels; returns the counts."""
    for sub in ('images', 'labels'):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    stats = Counter()
    before, after = Counter(), Counter()
    for image_path in sorted((src / 'images').iterdir()):
        stem = image_path.stem
        classes, boxes = read_labels(src / 'labels' / f'{stem}.txt')
        target = image_path.resolve()
        link(dst / 'images' / image_path.name, target)
        write_labels(dst / 'labels' / f'{stem}.txt', classes, boxes)
        stats['images'] += 1
        before.update(classes.tolist())
        after.update(classes.tolist())
        if not len(boxes) or not small_mask(boxes, args.small_px, args.ref_size).any():
            continue
        stats['small_images'] += 1
        for k in range(1, args.oversample):
            rng = random.Random(zlib.crc32(f'{args.seed}/{stem}/{k}'.encode()))
            new_classes, new_boxes, name = classes, boxes, f'{stem}_os{k}'
            if rng.random() < args.paste_prob:
                image = cv2.imread(str(target), cv2.IMREAD_COLOR)
                out, new_classes, new_boxes, pasted, failed = paste_small(image, classes, boxes, args, rng)
                stats['paste_failed'] += failed
                if pasted:
                    cv2.imwrite(str(dst / 'images' / f'{name}.png'), out)
                    stats['pasted_images'] += 1
                    stats['pasted_objects'] += pasted
                    after.update(new_classes[len(classes):].tolist())
                    write_labels(dst / 'labels' / f'{name}.txt', new_classes, new_boxes)
                    after.update(classes.tolist())
                    continue
            link(dst / 'images' / f'{name}{image_path.suffix}', target)  # a plain duplicate
            write_labels(dst / 'labels' / f'{name}.txt', classes, boxes)
            after.update(classes.tolist())
            stats['duplicate_images'] += 1
    stats['images_out'] = len(list((dst / 'images').iterdir()))
    return stats, before, after


def main():
    root = Path(__file__).resolve().parents[1] / 'dataset'
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, default=root / 'SARFact-YOLO', help='the output of prepare_yolo_data.py')
    p.add_argument('--dst', type=Path, default=root / 'SARFact-YOLO-aug')
    p.add_argument('--splits', nargs='+', default=['train'], choices=SPLITS,
                   help='splits to augment (val and test are linked as they are; give all three for the final model)')
    p.add_argument('--oversample', type=int, default=2, help='times an image with a small box appears (paper: 2-4, best 3)')
    p.add_argument('--paste-prob', type=float, default=0.5, help='chance that an extra copy is a copy-pasted one')
    p.add_argument('--paste-mode', choices=['multi', 'all'], default='multi',
                   help="multi: 1..max-objs small boxes, each 1..max-copies times; all: every small box once")
    p.add_argument('--max-objs', type=int, default=3)
    p.add_argument('--max-copies', type=int, default=3)
    p.add_argument('--small-px', type=float, default=32, help='small: box area < small-px^2 at --ref-size (COCO: 32)')
    p.add_argument('--ref-size', type=float, default=512, help='image size the detector sees')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.oversample < 1:
        p.error('--oversample: at least 1')
    src, dst = args.src.resolve(), args.dst.resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        p.error('--dst must be a folder of its own, outside --src')

    report = {}
    for split in SPLITS:
        out = dst / split
        if out.is_symlink():
            out.unlink()
        elif out.exists():
            shutil.rmtree(out)  # our own output of an earlier run
        if split in args.splits:
            stats, before, after = augment_split(src / split, out, args)
            report[split] = {**stats, 'boxes_before': dict(sorted(before.items())), 'boxes_after': dict(sorted(after.items()))}
            print(f"{split}: {stats['images']} -> {stats['images_out']} images "
                  f"({stats['small_images']} with small boxes, {stats['pasted_images']} pasted with {stats['pasted_objects']} "
                  f"objects, {stats['duplicate_images']} plain duplicates, {stats['paste_failed']} placements failed)")
            print(f"  boxes {sum(before.values())} -> {sum(after.values())}; per class {dict(sorted(after.items()))}")
        else:
            dst.mkdir(parents=True, exist_ok=True)
            out.symlink_to(src / split)
            print(f'{split}: linked to {src / split}')
    (dst / 'augment_log.json').write_text(json.dumps({'args': {k: str(v) for k, v in vars(args).items()}, **report}, indent=1))

    names = yaml.safe_load((src / 'data.yaml').read_text())['names']
    data = {'path': str(dst), 'train': 'train/images', 'val': 'val/images', 'test': 'test/images', 'names': names}
    (dst / 'data_aug.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    full = {**data, 'train': [f'{s}/images' for s in SPLITS]}
    (dst / 'data_full_aug.yaml').write_text(yaml.safe_dump(full, sort_keys=False))
    print(f'wrote {dst / "data_aug.yaml"} and {dst / "data_full_aug.yaml"}')


if __name__ == '__main__':
    main()
