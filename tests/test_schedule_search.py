"""Decisions and process handoff for the bounded, unattended BP7 search."""
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_bp7_schedule_search as search


def test_small_lr_changes_keep_the_existing_recipe():
    assert search.choose_lr_candidate({'lr1x': 50., 'hybrid130': 50.8, 'lr05x': 50.9}) == 'lr1x'


def test_boundary_is_a_tie_not_evidence_of_improvement():
    assert search.choose_lr_candidate({'lr1x': 50., 'hybrid130': 51., 'lr05x': 50.5}) == 'lr1x'
    assert search.choose_duration('hybrid130', 51., 52.) == 'hybrid130'


def test_clear_gain_can_select_new_learning_rates():
    assert search.choose_lr_candidate({'lr1x': 50., 'hybrid130': 51.5, 'lr05x': 50.2}) == 'hybrid130'
    assert search.choose_lr_candidate({'lr1x': 50., 'hybrid130': 50.8, 'lr05x': 52.}) == 'lr05x'


def test_near_best_hybrid_is_preferred_to_halving_both_groups():
    assert search.choose_lr_candidate({'lr1x': 50., 'hybrid130': 51.2, 'lr05x': 52.}) == 'hybrid130'


def test_duration_requires_a_gain_and_can_reject_longer_training():
    assert search.choose_duration('lr1x', 50., 51.01) == 'schedule160'
    assert search.choose_duration('lr1x', 50., 49.) == 'lr1x'


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_invalid_scores_cannot_drive_automatic_training(bad):
    with pytest.raises(ValueError):
        search.choose_lr_candidate({'lr1x': 50., 'hybrid130': bad, 'lr05x': 50.})
    with pytest.raises(ValueError):
        search.choose_duration('lr1x', 50., bad)


def test_long_trial_is_fresh_and_changes_the_full_schedule(tmp_path):
    cmd = search.build_training_command('/python', tmp_path/'data', tmp_path/'suite',
                                        'schedule160', 1e-4, 2e-4, 160)
    def value(flag):
        assert cmd.count(flag) == 1
        return cmd[cmd.index(flag) + 1]
    assert value('--epochs') == '160'
    assert value('--schedule-epochs') == '160'
    assert float(value('--lr')) == 1e-4
    assert float(value('--backbone-lr')) == 2e-4
    assert value('--train-split') == 'train'
    assert value('--eval-splits') == 'val'
    assert value('--batch-size') == '16'
    assert value('--workers') == '2'
    assert value('--seed') == '0'
    assert value('--warmup-epochs') == '1'
    assert '--resume' not in cmd and '--limit' not in cmd
    assert '--reproducible-loader' in cmd and '--save-training-state' in cmd


@pytest.mark.parametrize('stage', ['training', 'evaluating_submission_post', 'trial_complete'])
def test_active_dependency_never_starts_another_gpu_job(stage):
    assert not search.dependency_ready({'stage': stage}, locked=True, alive=True)


def test_complete_dependency_must_release_its_lock():
    assert not search.dependency_ready({'stage': 'complete'}, locked=True, alive=True)
    assert search.dependency_ready({'stage': 'complete'}, locked=False, alive=False)


def test_failed_or_dead_dependency_stops_the_search():
    with pytest.raises(RuntimeError):
        search.dependency_ready({'stage': 'failed'}, locked=False, alive=False)
    with pytest.raises(RuntimeError):
        search.dependency_ready({'stage': 'training'}, locked=False, alive=False)


@pytest.fixture
def valid_result_files(tmp_path):
    import json
    import train
    from evaluate_lr_checkpoint import local_delta_m, postprocessing_arguments
    data, run = tmp_path/'data', tmp_path/'trial'
    (data/'detection').mkdir(parents=True)
    run.mkdir()
    gt = data/'detection/instances_val.json'
    gt.write_text('{}')
    reference = tmp_path/'reference.json'
    ref_scores = {'macro_f1': .9, 'mAP': .2}
    reference.write_text(json.dumps({'val': ref_scores}))
    weights = run/'model.pt'
    weights.write_bytes(b'checkpoint identity fixture; no tensor loading in this validator')
    cmd = search.build_training_command('/python', data, run.parent, run.name, 1e-4, 2e-4, 130)
    args = vars(train.parse_args(cmd[cmd.index('train.py') + 1:]))
    args = json.loads(json.dumps(args, default=str))
    (run/'metrics.json').write_text(json.dumps({'args': args, 'parameters': 125076960,
                                              'trainable_parameters': 39867360, 'buffer_values': 16}))
    scores = {'accuracy': .95, 'macro_f1': .94, 'mAP': .4, 'AP50': .7}
    result = {'training_args': args, 'split': 'val', 'n_images': 940, 'training_splits': ['train'],
              'data': str(data), 'run': str(run), 'post': vars(postprocessing_arguments()),
              'weights': str(weights), 'weights_sha256': search.sha256_file(weights),
              'ground_truth_sha256': search.sha256_file(gt),
              'reference': {'path': str(reference), 'sha256': search.sha256_file(reference),
                            'split': 'val', **ref_scores},
              'metrics': scores, 'local_delta_m': local_delta_m(scores, ref_scores)}
    path = run/'metrics_submission_post_val.json'
    path.write_text(json.dumps(result))
    return run, path, data, reference, result


def test_accept_matching_result_and_reject_changed_checkpoint(valid_result_files):
    run, path, data, reference, _ = valid_result_files
    kwargs = dict(data=data, reference=reference, epochs=130, head_lr=1e-4, backbone_lr=2e-4)
    assert search.validate_result(run, path, **kwargs)['n_images'] == 940
    (run/'model.pt').write_bytes(b'wrong checkpoint')
    with pytest.raises(ValueError, match='checksum'):
        search.validate_result(run, path, **kwargs)


@pytest.mark.parametrize('mutation', ['coverage', 'post', 'score', 'training_data'])
def test_reject_incomparable_or_corrupted_scores(valid_result_files, mutation):
    import json
    run, path, data, reference, result = valid_result_files
    if mutation == 'coverage': result['n_images'] = 939
    if mutation == 'post': result['post']['shift'] = 0.
    if mutation == 'score': result['local_delta_m'] += .1
    if mutation == 'training_data': result['training_splits'] = ['train', 'val']
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        search.validate_result(run, path, data=data, reference=reference,
                               epochs=130, head_lr=1e-4, backbone_lr=2e-4)
