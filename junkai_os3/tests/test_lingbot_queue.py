"""CPU checks for the BP7 export-to-LingBot handoff and declared model budget."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_bp7_full_stop130 as bp7
import run_lingbot_full_stop130 as queued


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_full_command_changes_backbone_and_disables_incompatible_compile(tmp_path):
    args = ('/python', tmp_path / 'data', tmp_path / 'suite')
    original = bp7.training_command(*args)
    candidate = queued.training_command(*args)
    expected = list(original)
    expected[expected.index('--backbone') + 1] = 'lingbot_b'
    expected.remove('--compile')
    assert candidate == expected
    assert '--compile' in original and '--compile' not in candidate
    assert '--grad-ckpt' in candidate
    assert original[original.index('--backbone') + 1] == 'dinov3'
    assert candidate[candidate.index('--epochs') + 1] == '160'
    assert candidate[candidate.index('--schedule-epochs') + 1] == '160'
    assert candidate[candidate.index('--stop-after-epoch') + 1] == '130'


@pytest.fixture
def previous(tmp_path, monkeypatch):
    folder = tmp_path / 'previous'
    folder.mkdir()
    (folder / 'driver.lock').touch()
    output = folder / 'submission.zip'
    output.write_bytes(b'already validated predecessor archive')
    checkpoint = dict(epoch=130, global_step=76310, training_images=9392,
                      checkpoint_sha256='saved checkpoint digest')
    exported = dict(epoch=130, prediction_images=1426, format_passed=True,
                    checkpoint_sha256=checkpoint['checkpoint_sha256'], path=str(output),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    write_json(folder / 'checkpoint_validation.json', checkpoint)
    write_json(folder / 'submission_validation.json', exported)
    write_json(folder / 'status.json', dict(stage='complete', pid=123))
    identity = dict(command=[str(folder / 'code/tools/run_bp7_full_stop130.py')])
    monkeypatch.setattr(queued, 'process_identity', lambda pid: copy.deepcopy(identity))
    return folder


def test_running_predecessor_cannot_pass_the_handoff(previous, monkeypatch):
    write_json(previous / 'status.json', dict(stage='generating_submission', pid=123))
    updates = []

    class StillWaiting(Exception):
        pass

    def sleep(seconds):
        assert seconds == 30
        raise StillWaiting

    monkeypatch.setattr(queued.time, 'sleep', sleep)
    with pytest.raises(StillWaiting):
        queued.wait_for_previous(previous, lambda **item: updates.append(item))
    assert updates == [dict(dependency_stage='generating_submission')]


def test_complete_status_still_waits_for_lock_release(previous, monkeypatch):
    updates = []
    held = (previous / 'driver.lock').open('r')
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release_on_wait(seconds):
        assert updates == [dict(dependency_stage='complete')]
        fcntl.flock(held, fcntl.LOCK_UN)

    monkeypatch.setattr(queued.time, 'sleep', release_on_wait)
    try:
        result = queued.wait_for_previous(previous, lambda **item: updates.append(item))
    finally:
        held.close()
    assert result['previous_submission'] == str(previous / 'submission.zip')


@pytest.mark.parametrize('stage', ['failed', 'stopped_by_user', 'superseded'])
def test_failed_or_cancelled_dependency_aborts(previous, monkeypatch, stage):
    write_json(previous / 'status.json', dict(stage=stage, pid=123))
    monkeypatch.setattr(queued.time, 'sleep', lambda _: pytest.fail('Must abort before waiting'))
    with pytest.raises(RuntimeError, match='did not finish normally'):
        queued.wait_for_previous(previous, lambda **_: None)


def test_dead_running_dependency_aborts(previous, monkeypatch):
    write_json(previous / 'status.json', dict(stage='training', pid=123))
    monkeypatch.setattr(queued, 'process_identity', lambda _: None)
    with pytest.raises(RuntimeError, match='died before completion'):
        queued.wait_for_previous(previous, lambda **_: None)


def test_completed_dead_driver_can_pass_verified_handoff(previous, monkeypatch):
    monkeypatch.setattr(queued, 'process_identity', lambda _: None)
    monkeypatch.setattr(queued.time, 'sleep', lambda _: pytest.fail('Already complete and unlocked'))
    result = queued.wait_for_previous(previous, lambda **_: pytest.fail('Must not wait'))
    expected = hashlib.sha256((previous / 'submission.zip').read_bytes()).hexdigest()
    assert result['previous_submission_sha256'] == expected


@pytest.mark.parametrize('fault', ['wrong_epoch', 'wrong_steps', 'wrong_images',
                                   'unvalidated_export', 'different_checkpoint',
                                   'wrong_prediction_coverage', 'changed_zip', 'missing_record'])
def test_completed_dependency_rejects_invalid_artifacts(previous, fault):
    checkpoint = json.loads((previous / 'checkpoint_validation.json').read_text())
    exported = json.loads((previous / 'submission_validation.json').read_text())
    if fault == 'wrong_epoch':
        checkpoint['epoch'] = 120
    elif fault == 'wrong_steps':
        checkpoint['global_step'] -= 1
    elif fault == 'wrong_images':
        checkpoint['training_images'] = 7512
    elif fault == 'unvalidated_export':
        exported['format_passed'] = False
    elif fault == 'different_checkpoint':
        exported['checkpoint_sha256'] = 'another checkpoint'
    elif fault == 'wrong_prediction_coverage':
        exported['prediction_images'] = 1425
    elif fault == 'changed_zip':
        (previous / 'submission.zip').write_bytes(b'changed after validation')
    write_json(previous / 'checkpoint_validation.json', checkpoint)
    write_json(previous / 'submission_validation.json', exported)
    if fault == 'missing_record':
        (previous / 'checkpoint_validation.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        queued.wait_for_previous(previous, lambda **_: pytest.fail('Invalid artifacts must abort'))


def test_shared_stopped_validator_accepts_only_declared_budget(tmp_path):
    import torch
    import train

    run = tmp_path / 'full_train'
    run.mkdir()
    command = bp7.training_command('/python', tmp_path / 'data', tmp_path)
    command[command.index('--stop-after-epoch') + 1] = '1'
    args = vars(train.parse_args(command[command.index('train.py') + 1:]))
    args = json.loads(json.dumps(args, default=str))
    (run / 'log.txt').write_text(json.dumps(args) + '\n'
        'parameters: 4 total, 2 trainable (+ 1 floating-point buffer values)\n'
        'Stopped cleanly after epoch 1; resume from training_state.pt\n')
    trained = torch.tensor([1.0, 2.0])
    state = dict(format_version=1, epoch=1, global_step=3, scheduler=dict(last_epoch=3),
                 history=[], ema=None, rank_rng=[{}, {}], train_hours=0.01,
                 signature=dict(world_size=2, dataset_size=48, steps_per_epoch=3,
                                args=args, manifests={}, parameters={
                                    'frozen': dict(shape=[2], trainable=False),
                                    'trained': dict(shape=[2], trainable=True)}),
                 model=dict(frozen=torch.zeros(2), trained=trained, persistent=torch.ones(1)),
                 optimizer=dict(state={0: dict(step=torch.tensor(3.0))},
                                param_groups=[dict(initial_lr=1e-4, lr=1e-4)]))
    torch.save(state, run / 'training_state.pt')
    torch.save(dict(trained=trained, persistent=torch.ones(1)), run / 'ckpt_e1.pt')
    with pytest.raises(ValueError, match='Declared parameter budget mismatch'):
        bp7.verify_stopped_run(run, command, 1, 48)
    result = bp7.verify_stopped_run(run, command, 1, 48, expected_budget=(4, 2, 1))
    assert result['parameters'] == 4 and result['trainable_parameters'] == 2
    assert result['buffer_values'] == result['persistent_buffer_values'] == 1
    assert result['global_step'] == 3 and result['trained_tensors_match']


def test_lingbot_validator_uses_declared_budget_for_smoke_and_full(tmp_path, monkeypatch):
    suite = tmp_path / 'suite'
    (suite / 'validation').mkdir(parents=True)
    expected = (125105376, 39867360, 16)
    write_json(suite / 'launch_manifest.json', dict(expected_parameters=expected[0],
               expected_trainable_parameters=expected[1], expected_floating_buffers=expected[2]))
    calls = []

    def validate(run, command, epoch, images, *, expected_budget):
        calls.append((run, epoch, images, expected_budget))
        return dict(ok=True)

    monkeypatch.setattr(queued, 'verify_bp7_stopped_run', validate)
    queued.verify_stopped_run(suite / 'validation/smoke', [], 1, 48)
    queued.verify_stopped_run(suite / 'full_train', [], 130, 9392)
    assert calls == [(suite / 'validation/smoke', 1, 48, expected),
                     (suite / 'full_train', 130, 9392, expected)]
