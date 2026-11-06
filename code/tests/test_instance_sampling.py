"""Behavioral checks for GT-balanced Cascade RoI sampling."""

import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from sarbench.instance_sampling import (
    InstanceBalancedCascadeRoIHeads,
    instance_balanced_indices,
)
from sarbench.tiny import CascadeRoIHeads


def test_positives_are_round_robin_across_matched_objects():
    # Uniform positive sampling often misses the object with only two proposals.
    labels = torch.tensor([1] * 40 + [2] * 2 + [0] * 12)
    matched_gt = torch.tensor([0] * 40 + [1] * 2 + [0] * 12)

    selected = instance_balanced_indices(labels, matched_gt, batch_size=8, positive_fraction=0.5)
    chosen_positive_gt = matched_gt[selected[labels[selected] > 0]]

    assert torch.bincount(chosen_positive_gt, minlength=2).tolist() == [2, 2]
    assert selected.tolist() == sorted(selected.tolist())
    assert selected.unique().numel() == selected.numel()


def test_sampling_is_reproducible_from_torch_rng_state():
    labels = torch.tensor([1] * 20 + [2] * 9 + [0] * 80)
    matched_gt = torch.tensor([0] * 20 + [1] * 9 + [0] * 80)

    torch.manual_seed(174)
    first = instance_balanced_indices(labels, matched_gt, batch_size=12, positive_fraction=0.5)
    torch.manual_seed(174)
    second = instance_balanced_indices(labels, matched_gt, batch_size=12, positive_fraction=0.5)

    assert torch.equal(first, second)


def test_partial_round_robin_reaches_later_gt_ids_across_torch_seeds():
    labels = torch.ones(24, dtype=torch.int64)
    matched_gt = torch.arange(6).repeat_interleave(4)
    selected_gt = set()

    for seed in range(32):
        torch.manual_seed(seed)
        selected = instance_balanced_indices(labels, matched_gt, batch_size=2, positive_fraction=1.0)
        selected_gt.update(matched_gt[selected].tolist())

    assert selected_gt == set(range(6))


def test_sample_budget_ignores_ignored_rows_and_fills_with_background():
    labels = torch.tensor([1, 1, 2, 0, 0, -1, -1, 0])
    matched_gt = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])

    selected = instance_balanced_indices(labels, matched_gt, batch_size=6, positive_fraction=0.5)

    assert selected.numel() == 6  # three positives and all three available negatives
    assert (labels[selected] > 0).sum().item() == 3
    assert (labels[selected] == 0).sum().item() == 3
    assert (labels[selected] < 0).sum().item() == 0


def test_background_only_zero_positive_and_empty_inputs_are_supported():
    no_gt_labels = torch.zeros(5, dtype=torch.int64)
    no_gt = instance_balanced_indices(
        no_gt_labels, torch.zeros_like(no_gt_labels), batch_size=3, positive_fraction=0.25
    )
    no_positive_labels = torch.tensor([-1, -1, 0])
    no_positive = instance_balanced_indices(
        no_positive_labels, torch.zeros_like(no_positive_labels), batch_size=4, positive_fraction=0.5
    )
    no_background_labels = torch.tensor([1, 2, 1])
    no_background = instance_balanced_indices(
        no_background_labels, torch.tensor([0, 1, 0]), batch_size=4, positive_fraction=0.5
    )
    no_candidates = instance_balanced_indices(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64),
        batch_size=4, positive_fraction=0.25,
    )

    assert no_gt.numel() == 3 and (no_gt_labels[no_gt] == 0).all()
    assert no_positive.tolist() == [2]
    assert no_background.numel() == 2 and (no_background_labels[no_background] > 0).all()
    assert no_candidates.numel() == 0


def _build_heads(head_type, pool=None, **kwargs):
    return head_type(
        pool or MultiScaleRoIAlign(["0"], output_size=7, sampling_ratio=2),
        num_classes=3,
        reduced=1,
        representation=16,
        **kwargs,
    )


def test_subclass_has_no_new_state_and_eval_matches_original_cascade():
    torch.manual_seed(4)
    original = _build_heads(CascadeRoIHeads).eval()
    balanced = _build_heads(InstanceBalancedCascadeRoIHeads).eval()
    balanced.load_state_dict(original.state_dict())

    assert list(balanced.state_dict()) == list(original.state_dict())
    assert sum(p.numel() for p in balanced.parameters()) == sum(p.numel() for p in original.parameters())

    features = {"0": torch.randn(1, 256, 16, 16)}
    proposals = [torch.tensor([[1., 1., 8., 8.], [4., 3., 14., 12.], [10., 9., 20., 21.]])]
    image_shapes = [(32, 32)]
    with torch.no_grad():
        expected, expected_losses = original(features, proposals, image_shapes, return_raw=True)
        actual, actual_losses = balanced(features, proposals, image_shapes, return_raw=True)

    assert expected_losses == actual_losses == {}
    assert expected[0].keys() == actual[0].keys()
    for key in expected[0]:
        assert torch.equal(expected[0][key], actual[0][key])


class _RecordingPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.proposals_seen = []

    def forward(self, features, proposals, image_shapes):
        self.proposals_seen.append([p.detach().clone() for p in proposals])
        return features["0"].new_zeros((sum(len(p) for p in proposals), 256, 7, 7))


def test_cascade_drops_only_sampled_appended_gt_rows_between_all_three_stages():
    torch.manual_seed(22)
    pool = _RecordingPool()
    heads = _build_heads(
        InstanceBalancedCascadeRoIHeads,
        pool=pool,
        batch_size_per_image=6,
        positive_fraction=0.5,
    ).train()
    sampled_rows = []
    original_subsample = heads.subsample

    def record_subsample(labels):
        selected = original_subsample(labels)
        sampled_rows.append(([idx.clone() for idx in heads._sampled], list(heads._n_props)))
        return selected

    heads.subsample = record_subsample
    features = {"0": torch.randn(1, 256, 8, 8)}
    proposals = [torch.tensor([
        [0., 0., 5., 5.], [1., 1., 6., 6.], [8., 8., 13., 13.], [9., 9., 14., 14.],
        [18., 18., 23., 23.], [20., 20., 25., 25.], [26., 1., 30., 5.], [1., 26., 5., 30.],
    ])]
    targets = [{
        "boxes": torch.tensor([[2., 2., 7., 7.], [19., 19., 24., 24.]]),
        "labels": torch.tensor([1, 2]),
    }]

    _, losses = heads(features, proposals, [(32, 32)], targets)

    assert len(pool.proposals_seen) == len(sampled_rows) == 3
    assert set(losses) == {
        "loss_classifier_s0", "loss_box_reg_s0",
        "loss_classifier_s1", "loss_box_reg_s1",
        "loss_classifier_s2", "loss_box_reg_s2",
    }
    for stage, (indices, _) in enumerate(sampled_rows):
        assert indices[0].tolist() == sorted(indices[0].tolist())
        assert len(pool.proposals_seen[stage][0]) == len(indices[0])
    for stage in (0, 1):
        indices, n_original = sampled_rows[stage]
        appended_gt_sampled = (indices[0] >= n_original[0]).sum().item()
        assert sampled_rows[stage + 1][1][0] == len(indices[0]) - appended_gt_sampled

