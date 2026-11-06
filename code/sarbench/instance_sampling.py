"""Instance-balanced sampling for the copied Cascade R-CNN implementation."""

import torch

from .tiny import CascadeRoIHeads


def instance_balanced_indices(labels, matched_gt, batch_size, positive_fraction):
    """Return sorted sampled proposal rows with positives selected round-robin by matched GT.

    Positives within each GT bucket and background proposals are randomized with PyTorch's
    active RNG, so normal training checkpoints retain the usual save/resume reproducibility.
    Labels below zero are ignored, zero is background, and labels above zero are foreground.
    """
    if labels.ndim != 1 or matched_gt.ndim != 1 or labels.shape != matched_gt.shape:
        raise ValueError("labels and matched_gt must be one-dimensional tensors of equal length")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("positive_fraction must be between zero and one")

    positive_budget = int(batch_size * positive_fraction)
    positive_indices = torch.where(labels > 0)[0]
    num_positive = min(positive_indices.numel(), positive_budget)

    if num_positive:
        # Randomize proposal order first. A stable grouping then keeps that random order
        # within each GT while allowing the rows to be interleaved one rank at a time.
        random_order = torch.randperm(positive_indices.numel(), device=positive_indices.device)
        randomized = positive_indices[random_order]
        matched = matched_gt[randomized]
        group_order = torch.argsort(matched, stable=True)
        grouped = randomized[group_order]
        grouped_gt = matched[group_order]

        positions = torch.arange(grouped.numel(), device=labels.device)
        starts_group = torch.ones_like(grouped_gt, dtype=torch.bool)
        starts_group[1:] = grouped_gt[1:] != grouped_gt[:-1]
        group_ids = starts_group.to(torch.int64).cumsum(dim=0) - 1
        group_starts = torch.where(starts_group, positions, torch.zeros_like(positions)).cummax(dim=0).values
        group_ranks = positions - group_starts
        # Assign a random priority to each bucket. The positive-row count is a safe
        # permutation size and avoids reading a CUDA scalar for the dynamic bucket count.
        group_priority = torch.randperm(grouped.numel(), device=labels.device)[group_ids]
        round_robin_order = torch.argsort(
            group_ranks * grouped.numel() + group_priority, stable=True
        )
        selected_positive = grouped[round_robin_order[:num_positive]]
    else:
        selected_positive = positive_indices[:0]

    negative_indices = torch.where(labels == 0)[0]
    num_negative = min(negative_indices.numel(), batch_size - num_positive)
    if num_negative:
        random_order = torch.randperm(negative_indices.numel(), device=negative_indices.device)
        selected_negative = negative_indices[random_order[:num_negative]]
    else:
        selected_negative = negative_indices[:0]

    return torch.cat((selected_positive, selected_negative)).sort().values


class InstanceBalancedCascadeRoIHeads(CascadeRoIHeads):
    """Original Cascade RoI heads with GT-balanced positive proposal sampling.

    The parent cascade still owns proposal assignment, heads, prediction, and inference.
    This subclass only carries the current proposals' matched-GT indices into its sampler.
    """

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_gt, labels = super().assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        self._matched_gt_indices = matched_gt
        return matched_gt, labels

    def subsample(self, labels):
        sampler = self.fg_bg_sampler
        self._sampled = [
            instance_balanced_indices(
                image_labels,
                image_matched_gt,
                sampler.batch_size_per_image,
                sampler.positive_fraction,
            )
            for image_labels, image_matched_gt in zip(labels, self._matched_gt_indices)
        ]
        return self._sampled
