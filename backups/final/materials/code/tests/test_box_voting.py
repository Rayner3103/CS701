"""CPU checks for conservative coordinate voting on one forward pass's proposals."""
import pytest
import torch

from tools.evaluate_box_voting import vote_boxes


def test_weighted_neighbor_moves_only_a_fraction_of_the_way():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    candidates = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]])
    result = vote_boxes(
        boxes, torch.tensor([1]), candidates, torch.tensor([1.0, 3.0]), torch.tensor([1, 1]),
        iou_threshold=0.8, alpha=0.25,
    )
    # IoU is 9/11; the weighted target moves x by 3/4 px, then alpha retains one quarter.
    torch.testing.assert_close(result, torch.tensor([[0.1875, 0.0, 10.1875, 10.0]]))


def test_other_classes_and_distant_same_class_boxes_cannot_vote():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    candidates = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.1, 10.0, 10.1],  # high IoU, wrong class
        [3.0, 0.0, 13.0, 10.0],  # correct class, IoU below 0.8
    ])
    result = vote_boxes(
        boxes, torch.tensor([1]), candidates, torch.tensor([0.1, 100.0, 100.0]),
        torch.tensor([1, 2, 1]), iou_threshold=0.8, alpha=0.25,
    )
    torch.testing.assert_close(result, boxes, rtol=0, atol=0)


def test_higher_iou_threshold_excludes_the_same_neighbor():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    result = vote_boxes(
        boxes, torch.tensor([1]), torch.tensor([[1.0, 0.0, 11.0, 10.0]]),
        torch.tensor([1.0]), torch.tensor([1]), iou_threshold=0.9, alpha=0.25,
    )
    torch.testing.assert_close(result, boxes, rtol=0, atol=0)


def test_alpha_zero_preserves_coordinates_exactly_without_mutation():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    result = vote_boxes(
        boxes, torch.tensor([1]), torch.tensor([[1.0, 0.0, 11.0, 10.0]]),
        torch.tensor([1.0]), torch.tensor([1]), iou_threshold=0.8, alpha=0.0,
    )
    assert torch.equal(result, boxes)
    assert result.data_ptr() != boxes.data_ptr()


def test_empty_candidate_pool_and_empty_selected_boxes_are_valid():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    empty_boxes = torch.empty((0, 4))
    empty_labels = torch.empty(0, dtype=torch.int64)
    result = vote_boxes(
        boxes, torch.tensor([1]), empty_boxes, torch.empty(0), empty_labels,
        iou_threshold=0.8, alpha=0.25,
    )
    assert torch.equal(result, boxes)
    assert result.data_ptr() != boxes.data_ptr()
    result = vote_boxes(
        empty_boxes, empty_labels, boxes, torch.tensor([1.0]), torch.tensor([1]),
        iou_threshold=0.8, alpha=0.25,
    )
    assert result.shape == (0, 4)


def test_voting_does_not_mutate_any_inputs():
    inputs = (
        torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([1]),
        torch.tensor([[1.0, 0.0, 11.0, 10.0]]), torch.tensor([0.8]), torch.tensor([1]),
    )
    before = tuple(value.clone() for value in inputs)
    result = vote_boxes(*inputs, iou_threshold=0.8, alpha=0.25)
    assert not torch.equal(result, inputs[0])
    for actual, expected in zip(inputs, before):
        assert torch.equal(actual, expected)


def test_zero_weight_neighbors_leave_box_unchanged():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    result = vote_boxes(
        boxes, torch.tensor([1]), torch.tensor([[1.0, 0.0, 11.0, 10.0]]),
        torch.tensor([0.0]), torch.tensor([1]), iou_threshold=0.8, alpha=0.25,
    )
    assert torch.equal(result, boxes)


@pytest.mark.parametrize("parameter,value", [
    ("alpha", -0.1), ("alpha", 1.1), ("alpha", float("nan")),
    ("iou_threshold", 0.0), ("iou_threshold", -0.1),
    ("iou_threshold", 1.1), ("iou_threshold", float("nan")),
])
def test_invalid_voting_parameters_are_rejected(parameter, value):
    config = {"iou_threshold": 0.8, "alpha": 0.25, parameter: value}
    boxes, labels = torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([1])
    with pytest.raises(ValueError):
        vote_boxes(boxes, labels, boxes, torch.tensor([1.0]), labels, **config)
