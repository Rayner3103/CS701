"""augment_small_objects.py on a small synthetic YOLO dataset: oversampling counts, copy-paste rules (no overlap, border
margin, labels on the pasted pixels, same class), determinism, and that the source and the val split stay untouched."""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import augment_small_objects as aug  # noqa: E402

SIZE = 128


def args(**kw):
    base = dict(oversample=2, paste_prob=1.0, paste_mode='multi', max_objs=3, max_copies=3, small_px=32, ref_size=SIZE,
                seed=0, dst=None, src=None, splits=['train'])
    base.update(kw)
    return argparse.Namespace(**base)


def make_image(path, boxes):
    """Noise image with a bright square in each box (cx cy w h normalised)."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 60, (SIZE, SIZE, 3), dtype=np.uint8)
    for cx, cy, w, h in boxes:
        x1, y1 = int((cx - w / 2) * SIZE), int((cy - h / 2) * SIZE)
        x2, y2 = int((cx + w / 2) * SIZE), int((cy + h / 2) * SIZE)
        img[y1:y2, x1:x2] = 255
    cv2.imwrite(str(path), img)
    return img


@pytest.fixture
def source(tmp_path):
    src = tmp_path / 'src'
    for split in aug.SPLITS:
        (src / split / 'images').mkdir(parents=True)
        (src / split / 'labels').mkdir(parents=True)
    small = [(0.2, 0.2, 0.1, 0.1), (0.7, 0.3, 0.1, 0.1)]       # 12.8 px: small at ref_size 128, small_px 32 -> area 164 < 1024
    big = [(0.5, 0.5, 0.6, 0.6)]                                # 77 px: not small
    cases = {'a': (small, 3), 'b': (big, 0), 'c': ([(0.1, 0.1, 0.1, 0.1)] + big, 3)}
    for split in aug.SPLITS:
        for name, (boxes, cls) in cases.items():
            make_image(src / split / 'images' / f'{name}.png', boxes)
            (src / split / 'labels' / f'{name}.txt').write_text(''.join(f'{cls} {a} {b} {c} {d}\n' for a, b, c, d in boxes))
    (src / 'data.yaml').write_text(yaml.safe_dump({'names': {0: 'x', 3: 'car'}}))
    return src


def run(source, tmp_path, **kw):
    dst = tmp_path / 'dst'
    a = args(**kw)
    stats, before, after = aug.augment_split(source / 'train', dst / 'train', a)
    return dst / 'train', stats, before, after


def boxes_px(label_file):
    classes, boxes = aug.read_labels(label_file)
    return classes, aug.to_xyxy(boxes, SIZE, SIZE)


def test_small_definition():
    boxes = np.array([[.5, .5, .05, .05], [.5, .5, .6, .6], [.5, .5, 32 / 512, 32 / 512], [.5, .5, 31.9 / 512, 31.9 / 512]])
    assert aug.small_mask(boxes, 32, 512).tolist() == [True, False, False, True]


def test_oversampling_only(source, tmp_path):
    out, stats, before, after = run(source, tmp_path, oversample=3, paste_prob=0.0)
    names = sorted(p.name for p in (out / 'images').iterdir())
    assert names == ['a.png', 'a_os1.png', 'a_os2.png', 'b.png', 'c.png', 'c_os1.png', 'c_os2.png']  # b has no small box
    assert stats['small_images'] == 2 and stats['pasted_images'] == 0 and stats['duplicate_images'] == 4
    for name in names:  # every image has a label file with the labels of its original
        assert (out / 'labels' / f'{Path(name).stem}.txt').exists()
    assert (out / 'labels' / 'a_os2.txt').read_text() == (out / 'labels' / 'a.txt').read_text()
    assert (out / 'images' / 'a_os1.png').is_symlink()


def test_oversample_one_changes_nothing(source, tmp_path):
    out, stats, *_ = run(source, tmp_path, oversample=1)
    assert sorted(p.stem for p in (out / 'images').iterdir()) == ['a', 'b', 'c']


def test_paste_rules(source, tmp_path):
    out, stats, before, after = run(source, tmp_path, oversample=2, paste_prob=1.0)
    assert stats['pasted_images'] == 2 and stats['pasted_objects'] > 0
    for stem in ('a', 'c'):
        classes0, orig = boxes_px(out / 'labels' / f'{stem}.txt')
        classes, new = boxes_px(out / 'labels' / f'{stem}_os1.txt')
        assert len(new) > len(orig)
        assert (new[:len(orig)] == orig).all() and (classes[:len(orig)] == classes0).all()
        assert (classes == classes0[0]).all()                      # the image class never changes
        added = new[len(orig):]
        assert added[:, :2].min() >= aug.MARGIN - 0.5 and added[:, 2:].max() <= SIZE - aug.MARGIN + 0.5
        for i in range(len(new)):                                  # no box shares area with another one
            for j in range(i + 1, len(new)):
                assert aug.overlap_area(new[i], new[j]) < 1.0, (stem, i, j)
        # the pasted pixels are really there: the bright source squares (255) reappear inside the new boxes
        img = cv2.imread(str(out / 'images' / f'{stem}_os1.png'))
        for x1, y1, x2, y2 in added:
            inside = img[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]
            assert (inside == 255).mean() > 0.5
    assert sum(after.values()) > sum(before.values())


def test_big_objects_are_never_pasted(source, tmp_path):
    out, *_ = run(source, tmp_path, paste_prob=1.0)
    _, orig = boxes_px(out / 'labels' / 'c.txt')
    _, new = boxes_px(out / 'labels' / 'c_os1.txt')
    areas = (new[len(orig):, 2] - new[len(orig):, 0]) * (new[len(orig):, 3] - new[len(orig):, 1])
    assert (areas < 0.1 * SIZE * SIZE).all()                       # only the small box was copied (the 77 px one is not)


def test_paste_all_mode_pastes_each_small_box_once(source, tmp_path):
    out, stats, *_ = run(source, tmp_path, paste_mode='all')
    assert stats['pasted_objects'] + stats['paste_failed'] == 3     # a: two small boxes, c: one


def test_deterministic(source, tmp_path):
    out1, *_ = run(source, tmp_path / 'one', paste_prob=1.0)
    out2, *_ = run(source, tmp_path / 'two', paste_prob=1.0)
    assert (out1 / 'labels' / 'a_os1.txt').read_text() == (out2 / 'labels' / 'a_os1.txt').read_text()
    assert (cv2.imread(str(out1 / 'images' / 'a_os1.png')) == cv2.imread(str(out2 / 'images' / 'a_os1.png'))).all()


def test_crowded_image_skips_without_overlap(tmp_path):
    src = tmp_path / 'src' / 'train'
    (src / 'images').mkdir(parents=True)
    (src / 'labels').mkdir()
    grid = [(x, y, 0.05, 0.05) for x in np.arange(0.075, 0.95, 0.12) for y in np.arange(0.075, 0.95, 0.12)]
    make_image(src / 'images' / 'g.png', grid)
    (src / 'labels' / 'g.txt').write_text(''.join(f'0 {a} {b} {c} {d}\n' for a, b, c, d in grid))
    stats, *_ = aug.augment_split(src, tmp_path / 'dst', args(paste_prob=1.0))
    _, new = boxes_px(tmp_path / 'dst' / 'labels' / 'g_os1.txt')
    for i in range(len(new)):
        for j in range(i + 1, len(new)):
            assert aug.overlap_area(new[i], new[j]) < 1.0


def test_transformed_patch_scale_and_rotation():
    rng = random.Random(0)
    image = np.full((100, 100, 3), 200, np.uint8)
    sizes = []
    for _ in range(100):
        patch, mask = aug.transformed_patch(image, (40, 40, 50, 50), rng)
        assert patch.shape[:2] == mask.shape and mask.max() == 255
        sizes.append(patch.shape[1])
    assert 7 <= min(sizes) and max(sizes) <= 15                    # 10 px x [0.8, 1.2], plus rotation growth <= ~1.26x


def test_main_writes_yamls_and_links_other_splits(source, tmp_path, monkeypatch):
    # prepare a full source: main() reads SPLITS, data.yaml
    dst = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['x', '--src', str(source), '--dst', str(dst), '--ref-size', str(SIZE)])
    aug.main()
    assert (dst / 'val').is_symlink() and (dst / 'test').is_symlink() and not (dst / 'train').is_symlink()
    data = yaml.safe_load((dst / 'data_aug.yaml').read_text())
    assert data['train'] == 'train/images' and data['val'] == 'val/images' and data['names'] == {0: 'x', 3: 'car'}
    full = yaml.safe_load((dst / 'data_full_aug.yaml').read_text())
    assert full['train'] == ['train/images', 'val/images', 'test/images']
    assert len(list((source / 'train' / 'images').iterdir())) == 3   # the source is untouched
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, 'argv', ['x', '--src', str(source), '--dst', str(source)])
        aug.main()
