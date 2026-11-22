"""CPU checks for leakage rejection and the submission-compatible LR comparison score."""
import copy
from types import SimpleNamespace

import pytest
import torch

from tools.evaluate_lr_checkpoint import (evaluate_checkpoint, local_delta_m, parse_args, postprocessing_arguments,
                                          validate_holdout)


@pytest.mark.parametrize('training', ['val', 'train+val', 'train+val+test'])
def test_rejects_checkpoint_that_trained_on_validation(training):
    with pytest.raises(ValueError, match='checkpoint training split'):
        validate_holdout({'train_split': training}, 'val')


def test_holdout_validation_does_not_silently_select_test():
    args = parse_args(['--run', 'trial', '--data', 'data', '--out', 'score.json'])
    assert args.split == 'val'
    assert (args.batch_size, args.workers) == (4, 2)
    assert validate_holdout({'train_split': 'train'}, args.split) == ['train']
    with pytest.raises(ValueError):
        validate_holdout({'train_split': 'train+test'}, 'test')


def test_relative_score_and_fraction_guard():
    reference = {'macro_f1': 0.8, 'mAP': 0.4}
    assert local_delta_m(reference, reference) == 0
    assert local_delta_m({'macro_f1': 0.88, 'mAP': 0.48}, reference) == pytest.approx(15)
    with pytest.raises(ValueError, match='finite fraction'):
        local_delta_m({'macro_f1': 88, 'mAP': 48}, reference)
    with pytest.raises(ValueError, match='positive'):
        local_delta_m(reference, {'macro_f1': 0, 'mAP': 0.4})


def test_submission_post_is_reused_with_explicit_shift():
    from pp_eval import FINAL, LAM
    from predict import set_post
    before = copy.deepcopy(FINAL)
    args = postprocessing_arguments()
    model = SimpleNamespace(roi_heads=SimpleNamespace(post={}))
    set_post(model, args)
    for key, value in FINAL.items():
        assert model.roi_heads.post[key] == (-0.2 if key == 'shift' else value)
    assert args.cls_lambda == LAM
    assert FINAL == before


def test_submission_classification_uses_detector_fusion():
    from predict import image_class
    logits = torch.tensor([1.0, 0.0])  # the classifier alone chooses class 0
    output = {'raw_logits': torch.tensor([[0.0, -5.0, 5.0]])}
    assert image_class(logits, output, 0) == 0
    assert image_class(logits, output, postprocessing_arguments().cls_lambda) == 1


def test_evaluation_rejects_leakage_before_loading_weights_or_gpu(tmp_path, monkeypatch):
    import predict
    monkeypatch.setattr(predict, 'saved_args', lambda run: {'train_split': 'train+val+test'})
    def forbidden_load(*args, **kwargs):
        raise AssertionError('Model loading must not happen for a contaminated checkpoint')
    monkeypatch.setattr(predict, 'load_run', forbidden_load)
    args = parse_args(['--run', str(tmp_path), '--data', str(tmp_path),
                       '--out', str(tmp_path / 'scores.json')])
    with pytest.raises(ValueError, match='checkpoint training split'):
        evaluate_checkpoint(args)
    assert not args.out.exists()
