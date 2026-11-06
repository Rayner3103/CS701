"""CPU-only contract tests for opt-in native-size course geometry."""
import csv
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.ops import MultiScaleRoIAlign

from sarbench import data as data_module
from sarbench.data import SARImages, SARMultiTask, to_original_xywh
from sarbench.model import MultiTaskModel, configure_native_geometry


def make_course(root, *, width=20, height=10, box=(2, 2, 6, 6), label=1):
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "sample.png"
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).rectangle((box[0], box[1], box[0] + box[2] - 1, box[1] + box[3] - 1), fill=255)
    image.save(image_path)
    with (root / "train.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerow({"image": "sample.png", "label": label - 1})
    (root / "instances_train.json").write_text(json.dumps({
        "images": [{"id": 7, "file_name": "sample.png", "height": height, "width": width}],
        "annotations": [{"id": 9, "image_id": 7, "category_id": label, "bbox": list(box)}],
    }))
    (root / "classification").mkdir(exist_ok=True)
    (root / "detection").mkdir(exist_ok=True)
    (root / "classification" / "train.csv").write_text((root / "train.csv").read_text())
    (root / "detection" / "instances_train.json").write_text((root / "instances_train.json").read_text())
    return root


def test_default_dataset_keeps_legacy_stretched_resize(tmp_path):
    root = make_course(tmp_path / "course", width=20, height=10, box=(2, 2, 6, 4))
    _, _, target = SARMultiTask(root, "train", mean=0.0, std=1.0)[0]
    assert torch.allclose(target["boxes"], torch.tensor([[51.2, 102.4, 204.8, 307.2]]), atol=1e-4)
    assert "valid_size" not in target


def test_native_dataset_does_not_upscale_and_pads_after_normalization(tmp_path):
    root = make_course(tmp_path / "course", width=40, height=24, box=(5, 4, 10, 8))
    image, _, target = SARMultiTask(root, "train", native_size=True, mean=0.0, std=1.0)[0]
    assert image.shape == (1, 512, 512)
    assert target["orig_size"] == (24, 40)
    assert target["valid_size"] == (24, 40)
    assert target["scale_xy"] == (1.0, 1.0)
    assert torch.equal(target["boxes"], torch.tensor([[5.0, 4.0, 15.0, 12.0]]))
    assert image[0, 4:12, 5:15].min() == 1
    assert torch.count_nonzero(image[0, 24:, :]) == 0
    assert torch.count_nonzero(image[0, :, 40:]) == 0


def test_sar_images_native_pixels_and_geometry_match_labeled_path(tmp_path):
    root = make_course(tmp_path / "course", width=256, height=128, box=(25, 20, 50, 40))
    stats = {"mean": 0.2658373163, "std": 0.2054887053}
    labeled_image, _, target = SARMultiTask(root, "train", native_size=True, **stats)[0]
    unlabeled_image, metadata = SARImages(
        root, [{"id": 7, "file_name": "sample.png", "height": 128, "width": 256}],
        native_size=True, **stats,
    )[0]
    assert torch.equal(unlabeled_image, labeled_image)
    assert metadata["orig_size"] == target["orig_size"] == (128, 256)
    assert metadata["valid_size"] == target["valid_size"] == (128, 256)
    assert metadata["scale_xy"] == target["scale_xy"] == (1.0, 1.0)


def test_native_downscale_uses_actual_xy_scales_for_boxes_and_inverse(tmp_path):
    root = make_course(tmp_path / "course", width=801, height=401, box=(100, 80, 200, 100))
    _, _, target = SARMultiTask(root, "train", native_size=True, mean=0.0, std=1.0)[0]
    sx, sy = target["scale_xy"]
    assert target["valid_size"] == (256, 512)
    assert sx == pytest.approx(512 / 801)
    assert sy == pytest.approx(256 / 401)
    assert torch.allclose(target["boxes"], torch.tensor([[100 * sx, 80 * sy, 300 * sx, 180 * sy]]))
    recovered = to_original_xywh(target["boxes"], target["orig_size"], valid_size=target["valid_size"],
                                 scale_xy=target["scale_xy"])
    assert torch.allclose(recovered, torch.tensor([[100.0, 80.0, 200.0, 100.0]]), atol=1e-4)


def test_native_boxes_are_clipped_to_valid_region_before_augmentation(tmp_path):
    root = make_course(tmp_path / "course", width=40, height=24, box=(35, 20, 10, 10))
    _, _, target = SARMultiTask(root, "train", native_size=True, mean=0.0, std=1.0)[0]
    assert torch.equal(target["boxes"], torch.tensor([[35.0, 20.0, 40.0, 24.0]]))
    assert target["labels"].tolist() == [1]


def test_native_flip_uses_valid_extent_and_scales_bias_by_axis(tmp_path, monkeypatch):
    root = make_course(tmp_path / "course", width=40, height=24, box=(5, 4, 10, 8))
    draws = iter([0.0, 1.0])  # horizontal flip only
    monkeypatch.setattr(data_module.random, "random", lambda: next(draws))
    image, _, target = SARMultiTask(root, "train", train=True, aug="flip", native_size=True,
                                    flip_bias=1.0, mean=0.0, std=1.0)[0]
    assert torch.allclose(target["boxes"], torch.tensor([[27.0, 4.0, 37.0, 12.0]]))
    assert image[0, 4:12, 30:35].min() == 1
    assert torch.count_nonzero(image[0, 24:, :]) == 0
    assert torch.count_nonzero(image[0, :, 40:]) == 0
    assert len(target["boxes"]) == len(target["labels"]) == 1


def test_native_mode_rejects_zoom(tmp_path):
    root = make_course(tmp_path / "course")
    with pytest.raises(ValueError, match="zoom"):
        SARMultiTask(root, "train", train=True, native_size=True, zoom_p=0.1)


def test_native_d4_rotates_only_valid_region_and_swaps_valid_size(tmp_path, monkeypatch):
    root = make_course(tmp_path / "course", width=40, height=24, box=(5, 4, 10, 8))
    monkeypatch.setattr(data_module.random, "random", lambda: 1.0)
    monkeypatch.setattr(data_module.random, "randrange", lambda n: 1)
    image, _, target = SARMultiTask(root, "train", train=True, aug="d4", native_size=True, mean=0.0, std=1.0)[0]
    assert target["valid_size"] == (40, 24)
    assert torch.equal(target["boxes"], torch.tensor([[4.0, 25.0, 12.0, 35.0]]))
    assert torch.count_nonzero(image[0, 40:, :]) == 0
    assert torch.count_nonzero(image[0, :, 24:]) == 0
    assert image[0, 25:35, 4:12].min() == 1


def test_native_mean_pool_weights_partial_patches_and_ignores_prefix_tokens():
    model_like = SimpleNamespace(
        cls_pool="mean",
        backbone=SimpleNamespace(patch_size=16, token_stride=16, prefix_tokens=torch.full((1, 1, 1), 999.0)),
    )
    tokens = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    pooled = MultiTaskModel.pool(model_like, tokens, image_sizes=[(20, 24)], canvas_size=(32, 32))
    # Fractions are [[1, .5], [.25, .125]] for these four 16 px patches.
    expected = (1.0 + 2.0 * 0.5 + 3.0 * 0.25 + 4.0 * 0.125) / (1.0 + 0.5 + 0.25 + 0.125)
    assert pooled.item() == pytest.approx(expected)


def test_native_hierarchical_mean_pool_masks_partial_feature_cells():
    feature_map = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    pooled = MultiTaskModel._pool_valid_map(feature_map, [(20, 24)], (32, 32))
    expected = (1.0 + 2.0 * 0.5 + 3.0 * 0.25 + 4.0 * 0.125) / (1.0 + 0.5 + 0.25 + 0.125)
    assert pooled.item() == pytest.approx(expected)


def test_native_roi_pool_infers_stride_from_padded_canvas():
    original = MultiScaleRoIAlign(["0", "1", "2", "3"], output_size=2, sampling_ratio=2)
    model_like = SimpleNamespace(roi_heads=SimpleNamespace(box_roi_pool=original))
    configure_native_geometry(model_like, canvas_size=(512, 512))
    wrapped = model_like.roi_heads.box_roi_pool
    features = {
        "0": torch.ones(1, 1, 128, 128),
        "1": torch.ones(1, 1, 64, 64),
        "2": torch.ones(1, 1, 32, 32),
        "3": torch.ones(1, 1, 16, 16),
    }
    pooled = wrapped(features, [torch.tensor([[0.0, 0.0, 256.0, 256.0]])], [(256, 256)])
    assert pooled.shape == (1, 1, 2, 2)
    assert original.scales == [0.25, 0.125, 0.0625, 0.03125]


class PatchMeanBackbone(nn.Module):
    embed_dim = 128
    patch_size = 16

    def forward(self, images):
        patches = F.avg_pool2d(images, self.patch_size).flatten(2).transpose(1, 2)
        return patches.expand(-1, -1, self.embed_dim)


class CaptureRPN(nn.Module):
    last_scores = None

    def forward(self, image_list, features, targets=None):
        self.image_sizes_seen = image_list.image_sizes
        proposal = image_list.tensors.new_tensor([[0.0, 0.0, 128.0, 128.0]])
        return [proposal], {}


class CaptureROI(nn.Module):
    def forward(self, features, proposals, image_shapes, targets=None, image_probs=None, return_raw=False,
                objectness=None):
        self.image_shapes_seen = image_shapes
        empty = proposals[0].new_zeros((0, 4))
        return [{"boxes": empty, "scores": empty.new_zeros((0,)), "labels": empty.new_zeros((0,), dtype=torch.long)}], {}


def test_forward_masks_padded_pixels_and_passes_valid_sizes_to_detection_heads():
    model = MultiTaskModel(PatchMeanBackbone(), task_routing=False, box_head="small").eval()
    rpn, roi = CaptureRPN(), CaptureROI()
    model.rpn, model.roi_heads = rpn, roi
    configure_native_geometry(model, canvas_size=(512, 512))
    images = torch.full((1, 1, 512, 512), 100.0)
    images[:, :, :256, :256] = 1.0
    pooled = []
    model.cls_head.register_forward_pre_hook(lambda module, args: pooled.append(args[0].detach()))
    with torch.no_grad():
        model(images, image_sizes=[(256, 256)])
    assert torch.allclose(pooled[0], torch.ones_like(pooled[0]))
    assert rpn.image_sizes_seen == [(256, 256)]
    assert roi.image_shapes_seen == [(256, 256)]


def test_dinov3_can_use_local_timm_weights_without_importing_terratorch():
    script = textwrap.dedent(r"""
        import builtins, sys, types
        import torch.nn as nn
        original_import = builtins.__import__
        def forbid_terratorch(name, *args, **kwargs):
            if name.startswith('terratorch'):
                raise ModuleNotFoundError('terratorch intentionally unavailable')
            return original_import(name, *args, **kwargs)
        builtins.__import__ = forbid_terratorch
        timm = types.ModuleType('timm')
        timm.create_model = None
        sys.modules['timm'] = timm
        import sarbench.backbones as module
        from types import SimpleNamespace
        class FakeTimmModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim = 768
                self.patch_embed = SimpleNamespace(patch_size=(16, 16))
        calls = {}
        def create_model(*args, **kwargs):
            calls.update(args=args, kwargs=kwargs)
            return FakeTimmModel()
        timm.create_model = create_model
        backbone = module.build_backbone('dinov3', pretrained=True, pretrained_file='D:/weights/dinov3.pth')
        assert backbone.vit is not None
        assert calls['kwargs']['pretrained_cfg_overlay'] == {'file': 'D:/weights/dinov3.pth'}
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
