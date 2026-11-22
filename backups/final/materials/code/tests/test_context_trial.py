"""Context trial recipe and result identity checks; no GPU or model construction."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_bp7_context_trial as context
from run_bp7_schedule_search import build_training_command, sha256_file


def test_context_is_the_only_recipe_change(tmp_path):
    data, suite = tmp_path / 'data', tmp_path / 'suite'
    baseline = build_training_command('/python', data, suite, 'context160', 1e-4, 3e-4, 160)
    actual = context.training_command('/python', data, suite)
    assert actual == baseline + ['--roi-context', '3', '--roi-context-mode', 'classification']
    for flag, expected in {'--lr': '0.0001', '--backbone-lr': '0.0003',
                           '--epochs': '160', '--schedule-epochs': '160',
                           '--warmup-epochs': '1', '--seed': '0', '--batch-size': '16',
                           '--train-split': 'train', '--eval-splits': 'val'}.items():
        assert actual.count(flag) == 1
        assert actual[actual.index(flag) + 1] == expected
    assert '--resume' not in actual
    assert '--save-training-state' in actual and '--reproducible-loader' in actual


@pytest.fixture
def expected_args(tmp_path):
    import train
    command = context.training_command('/python', tmp_path / 'data', tmp_path / 'suite')
    args = vars(train.parse_args(command[command.index('train.py') + 1:]))
    return json.loads(json.dumps(args, default=str))


@pytest.fixture
def metrics(expected_args):
    return dict(args=copy.deepcopy(expected_args), parameters=125126304,
                trainable_parameters=39916704, buffer_values=16)


def test_accept_context_parameter_counts(metrics, expected_args):
    context.validate_training_metrics(metrics, expected_args)


@pytest.mark.parametrize('key,value', [
    ('parameters', 125076960),  # Original BP7 counts must not pass for context training.
    ('trainable_parameters', 39867360),
    ('parameters', 125126305),
    ('trainable_parameters', 40000001),
    ('buffer_values', 130000000 - 125126304 + 1),
])
def test_reject_wrong_architecture_or_budget(metrics, expected_args, key, value):
    metrics[key] = value
    with pytest.raises(ValueError, match='parameter|budget'):
        context.validate_training_metrics(metrics, expected_args)


@pytest.mark.parametrize('key,value', [
    ('roi_context_mode', 'shared'), ('roi_context', 0),
    ('epochs', 130), ('schedule_epochs', 130),
    ('train_split', 'train+val+test'), ('lr', 1.25e-4),
])
def test_reject_changed_training_recipe(metrics, expected_args, key, value):
    metrics['args'][key] = value
    with pytest.raises(ValueError, match='arguments'):
        context.validate_training_metrics(metrics, expected_args)


@pytest.fixture
def evaluation(expected_args, tmp_path):
    from evaluate_lr_checkpoint import postprocessing_arguments
    weight_file = tmp_path / 'model.pt'
    weight_file.write_bytes(b'context checkpoint fixture; no tensor loading')
    post = vars(postprocessing_arguments())
    result = dict(training_args=copy.deepcopy(expected_args),
                  weights_sha256=sha256_file(weight_file), post=copy.deepcopy(post),
                  split='val', n_images=940, training_splits=['train'])
    return result, weight_file, post


def test_accept_evaluation_and_reject_changed_checkpoint(evaluation, expected_args):
    result, weights, post = evaluation
    context.validate_evaluation(result, expected_args=expected_args,
                                weights_sha256=sha256_file(weights), post=post)
    weights.write_bytes(b'another checkpoint, same path')
    with pytest.raises(ValueError, match='hash'):
        context.validate_evaluation(result, expected_args=expected_args,
                                    weights_sha256=sha256_file(weights), post=post)


@pytest.mark.parametrize('mutation', ['post', 'split', 'coverage', 'training_splits', 'context_mode'])
def test_reject_incomparable_evaluation(evaluation, expected_args, mutation):
    result, weights, post = evaluation
    if mutation == 'post':
        result['post']['shift'] = 0
    elif mutation == 'split':
        result['split'] = 'test'
    elif mutation == 'coverage':
        result['n_images'] = 939
    elif mutation == 'training_splits':
        result['training_splits'] = ['train', 'val']
    else:
        result['training_args']['roi_context_mode'] = 'shared'
    with pytest.raises(ValueError):
        context.validate_evaluation(result, expected_args=expected_args,
                                    weights_sha256=sha256_file(weights), post=post)
