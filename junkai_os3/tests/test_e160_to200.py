"""CPU coverage of natural e200 completion, e180 observation and isolated cleanup."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_bp7_cos200_final as original
import run_bp7_e147_recovery as recovery
import run_bp7_e160_to200 as final
import train


def template(data, source):
    command = original.training_command('/python', data, source, 'full_train_e160', stop_epoch=160,
                                        resume=source / 'old/training_state.pt')
    command[command.index('--memory-fraction') + 1] = '0.35'
    return command


def args_for(command):
    return train.parse_args(command[command.index('train.py') + 1:])


def synthetic_state(command, epoch):
    args = args_for(command)
    config = json.loads(json.dumps({k: v for k, v in vars(args).items()
                                   if k not in recovery.IGNORED_ARGS}, default=str))
    steps = epoch * 587
    factor = train.lr_factor(steps, 587, 200 * 587)
    return dict(format_version=1, epoch=epoch, global_step=steps, scheduler={'last_epoch': steps},
        model={'p': torch.ones(2), 'buffer': torch.tensor([3.])}, ema=None, history=[], train_hours=20,
        optimizer=dict(state={0: {'step': torch.tensor(steps)}},
                       param_groups=[dict(initial_lr=1e-4, lr=1e-4 * factor)]),
        rank_rng=[dict(python=(), numpy=(), torch=torch.tensor([1]), cuda=torch.tensor([2])) for _ in range(2)],
        signature=dict(args=config, world_size=2, dataset_size=9392, steps_per_epoch=587,
                       manifests={}, parameters={'p': dict(shape=[2], trainable=True)}))


def test_continuation_preserves_all_training_configuration(tmp_path):
    old = template(tmp_path / 'data', tmp_path / 'source')
    new = final.continuation_command(old, tmp_path / 'suite', tmp_path / 'state.pt')
    before, after = vars(args_for(old)), vars(args_for(new))
    assert {k: v for k, v in before.items() if k not in recovery.IGNORED_ARGS} == {
        k: v for k, v in after.items() if k not in recovery.IGNORED_ARGS}
    assert after['stop_after_epoch'] == after['epochs'] == after['schedule_epochs'] == 200
    assert after['save_every'] == 10 and after['batch_size'] == 16
    assert after['resume'] == tmp_path / 'state.pt'
    assert old == template(tmp_path / 'data', tmp_path / 'source')


@pytest.mark.parametrize('fault', [None, 'epoch', 'lr', 'model', 'missing_buffer', 'metrics_args', 'budget', 'hours', 'completion'])
def test_final200_uses_model_and_metrics_not_stopped_checkpoint(tmp_path, fault):
    command = final.continuation_command(template(tmp_path / 'data', tmp_path / 'source'),
                                         tmp_path, tmp_path / 'source.pt')
    state = synthetic_state(command, 200)
    metrics = dict(args=json.loads(json.dumps(vars(args_for(command)), default=str)),
                   parameters=2, trainable_parameters=2, buffer_values=1,
                   val_history=[], train_hours=20)
    inference = {k: v.clone() for k, v in state['model'].items()}
    log = 'epoch 200 iter 587/587 loss 0.1\nRESULT {}\n'
    if fault == 'epoch': state['epoch'] = 199
    elif fault == 'lr': state['optimizer']['param_groups'][0]['lr'] = 1e-5
    elif fault == 'model': inference['p'][0] = 2
    elif fault == 'missing_buffer': del inference['buffer']
    elif fault == 'metrics_args': metrics['args']['lr'] = 2e-4
    elif fault == 'budget': metrics['trainable_parameters'] = 3
    elif fault == 'hours': metrics['train_hours'] = 21
    elif fault == 'completion': log = 'Stopped cleanly after epoch 200;\n'
    torch.save(state, tmp_path / 'training_state.pt')
    torch.save(inference, tmp_path / 'model.pt')
    final.write_json(tmp_path / 'metrics.json', metrics)
    (tmp_path / 'log.txt').write_text(log)
    if fault:
        with pytest.raises(ValueError):
            final.verify_final_run(tmp_path, command, budget=(2, 2, 1))
    else:
        checked = final.verify_final_run(tmp_path, command, budget=(2, 2, 1))
        assert checked['global_step'] == 117400 and checked['lr_factor'] == 0
        assert checked['checkpoint'] == str(tmp_path / 'model.pt')
        assert not (tmp_path / 'ckpt_e200.pt').exists()


def test_midpoint_verifies_same_inode_across_atomic_replace(tmp_path, monkeypatch):
    command = final.continuation_command(template(tmp_path / 'data', tmp_path / 'source'),
                                         tmp_path, tmp_path / 'source.pt')
    state = synthetic_state(command, 180)
    torch.save(state, tmp_path / 'training_state.pt')
    torch.save(state['model'], tmp_path / 'ckpt_e180.pt')
    replacement = tmp_path / 'next.pt'
    torch.save(synthetic_state(command, 181), replacement)
    original_verify = final.verify_state
    def replace_then_verify(path, command, epoch, **kwargs):
        assert epoch == 180
        os.replace(replacement, tmp_path / 'training_state.pt')
        return original_verify(path, command, epoch, **kwargs)
    monkeypatch.setattr(final, 'verify_state', replace_then_verify)
    checks = final.observe_checkpoint(tmp_path, command, {'first': True}, budget=(2, 2))
    assert checks['180']['global_step'] == 105660
    assert checks['180']['trained_tensors_match']
    assert torch.load(tmp_path / 'training_state.pt', weights_only=False)['epoch'] == 181


def test_missed_midpoint_is_explicit_and_does_not_validate_wrong_epoch(tmp_path):
    command = final.continuation_command(template(tmp_path / 'data', tmp_path / 'source'),
                                         tmp_path, tmp_path / 'source.pt')
    state = synthetic_state(command, 181)
    torch.save(state, tmp_path / 'training_state.pt')
    torch.save(state['model'], tmp_path / 'ckpt_e180.pt')
    checks = final.observe_checkpoint(tmp_path, command, {'first': True}, budget=(2, 2))
    assert set(checks) == {'180_skipped'} and checks['180_skipped']['observed_epoch'] == 181


def test_cleanup_only_owns_new_suite_and_rechecks_process_identity(tmp_path, monkeypatch):
    suite = tmp_path / 'new'
    own = ['python', 'train.py', '--out', str(suite)]
    infer = ['python', 'predict.py', '--run', str(suite / final.RUN_NAME)]
    other = ['llama-server', '--model', 'other']
    old = ['python', 'train.py', '--out', str(tmp_path / 'old')]
    table = {100: dict(group=100, start='1', argv=own),
             200: dict(group=200, start='2', argv=infer),
             300: dict(group=300, start='3', argv=other),
             400: dict(group=400, start='4', argv=own),
             401: dict(group=400, start='5', argv=other),
             500: dict(group=500, start='6', argv=old)}
    groups, pids = [], []
    monkeypatch.setattr(final, 'process_table', lambda: table)
    monkeypatch.setattr(final.os, 'getpgrp', lambda: 999)
    monkeypatch.setattr(final.os, 'killpg', lambda p, sig: groups.append((p, sig)))
    monkeypatch.setattr(final.os, 'kill', lambda p, sig: pids.append((p, sig)))
    final.kill_remaining_owned(suite)
    assert groups == [(100, signal.SIGKILL), (200, signal.SIGKILL)]
    assert pids == [(400, signal.SIGKILL)]
    assert not final.owned_command(['python', '--suite', str(suite)], suite)


@pytest.mark.parametrize('mode', ['normal', 'miss180', 'resource', 'stall', 'exit'])
def test_main_one_ddp_then_sequential_exports_and_watchdog(tmp_path, monkeypatch, mode):
    # A short HDD temp path also exercises the Unix socket path-length guard.
    root = Path(__file__).resolve().parents[1] / '.runtime'
    root.mkdir(exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix='t200-', dir=root))
    suite, source = base / 'new', base / 'old'
    for path in (suite / 'code/tools', source / 'full_train_e160'):
        path.mkdir(parents=True)
    (source / 'driver.lock').touch()
    checkpoint = source / 'full_train_e160/training_state.pt'
    checkpoint.write_bytes(b'protected e160')
    existing = base / 'e160.zip'
    existing.write_bytes(b'protected best')
    driver = suite / 'code/tools/run_bp7_e160_to200.py'
    driver.touch()
    command = template(base / 'data', source)
    outputs = {str(e): str(base / f'e{e}.zip') for e in (180, 200)}
    manifest = dict(source_suite=str(source), temp_dir=str(base / 't'),
                    existing_e160_submission=str(existing), submission_paths=outputs,
                    data=str(base / 'data'), official_data=str(base / 'official'))
    final.write_json(suite / 'launch_manifest.json', manifest)
    monkeypatch.setattr(final, '__file__', str(driver))
    monkeypatch.setattr(sys, 'argv', [str(driver), '--suite', str(suite)])
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
    for key in ('TMPDIR', 'TMP', 'TEMP', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        monkeypatch.setenv(key, 'wrong inherited value')
    monkeypatch.setattr(original, 'original_resources', lambda: None)
    monkeypatch.setattr(final, 'verify_inputs', lambda *a: (command, {'epoch': 160}))
    monkeypatch.setattr(final, 'verify_data', lambda *a: ({}, {'count': 9392}))
    monkeypatch.setattr(final, 'validate_zip', lambda p, images: {'sha256': final.sha256_file(p)})
    calls, stops, verified = [], [], []
    clock = [0.]
    monkeypatch.setattr(final.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(final, 'stop_child', lambda child, scope: stops.append((child.pid, scope)))
    def observe(run, cmd, observed):
        if mode in ('resource', 'stall', 'exit'):
            return {}
        if not observed:
            return {'first': {'epoch': 161}}
        if mode == 'miss180':
            return {'180_skipped': {'observed_epoch': 200}}
        return {'180': {'epoch': 180, 'checkpoint_sha256': final.sha256_file(run / 'ckpt_e180.pt')}}
    monkeypatch.setattr(final, 'observe_checkpoint', observe)
    def finish(run, cmd):
        assert len(calls) == 1 and (run / 'model.pt').exists()
        verified.append(200)
        return {'epoch': 200, 'checkpoint_sha256': final.sha256_file(run / 'model.pt'), 'train_hours': 20}
    monkeypatch.setattr(final, 'verify_final_run', finish)
    class Child:
        pid = 424242
        def __init__(self, cmd, cwd, env, start_new_session, stdout, stderr):
            assert start_new_session and cwd == suite / 'code'
            assert all(env[k] == manifest['temp_dir'] for k in ('TMPDIR', 'TMP', 'TEMP'))
            assert tempfile.gettempdir() == manifest['temp_dir']
            assert env['PYTORCH_CUDA_ALLOC_CONF'] == final.ALLOCATOR
            assert 'PYTORCH_ALLOC_CONF' not in env
            self.command, self.stream, self.count = cmd, stdout, 0
            self.training = 'train.py' in cmd
            calls.append(cmd)
            assert final.value(cmd, '--memory-fraction') == '0.35'
            if self.training:
                args = args_for(cmd)
                assert args.resume == checkpoint and args.stop_after_epoch == 200
                assert args.epochs == args.schedule_epochs == 200
                assert args.lr == 1e-4 and args.backbone_lr == 3e-4
                assert args.workers == 1 and args.batch_size == 16 and not args.compile
                self.run = args.out / args.name
            else:
                assert verified == [200] and env['CUDA_VISIBLE_DEVICES'] == '1'
                expected = 'model.pt' if len(calls) == 2 else 'ckpt_e180.pt'
                assert final.value(cmd, '--weights') == expected
                Path(final.value(cmd, '--out')).write_bytes(expected.encode())
        def wait(self, timeout):
            assert timeout == 30
            self.count += 1
            if not self.training:
                return 0
            if mode == 'resource':
                self.stream.write('CUDA out of memory\n')
            elif mode == 'stall':
                clock[0] += 100
                self.stream.write('worker warning\n')
            elif mode == 'exit':
                return 1
            else:
                epoch = 161 if self.count == 1 else 200
                self.stream.write(f'epoch {epoch} iter 587/587 loss 0.1\n')
                if self.count > 1:
                    (self.run / 'ckpt_e180.pt').write_bytes(b'e180 weights')
                    (self.run / 'model.pt').write_bytes(b'e200 weights')
                    self.stream.flush()
                    return 0
            self.stream.flush()
            raise subprocess.TimeoutExpired(self.command, timeout)
    monkeypatch.setattr(final.subprocess, 'Popen', Child)
    try:
        if mode in ('resource', 'stall', 'exit'):
            with pytest.raises((RuntimeError, TimeoutError, subprocess.CalledProcessError)):
                final.main()
        else:
            final.main()
        status = final.read_json(suite / 'status.json')
        assert checkpoint.read_bytes() == b'protected e160' and existing.read_bytes() == b'protected best'
        assert len([c for c in calls if 'train.py' in c]) == 1
        assert 'child_pid' not in status
        if mode in ('resource', 'stall', 'exit'):
            assert status['stage'] == 'failed' and len(calls) == 1
            assert stops == [(424242, suite)]
        else:
            assert status['stage'] == 'complete' and not stops
            assert Path(outputs['200']).with_suffix('.sha256').exists()
            assert Path(outputs['180']).exists() == (mode == 'normal')
            assert len(calls) == (3 if mode == 'normal' else 2)
            assert ('skipped_submission_e180' in status) == (mode == 'miss180')
    finally:
        shutil.rmtree(base)


@pytest.mark.parametrize('save', [None, 'trainable', 'none'])
def test_input_validation_accepts_implicit_save_and_rejects_wrong_effective_save(tmp_path, monkeypatch, save):
    source, suite = tmp_path / 'old', tmp_path / 'new'
    data, official = tmp_path / 'data', tmp_path / 'official'
    for path in (source / 'code', source / 'full_train_e160', suite / 'code/tools'):
        path.mkdir(parents=True)
    for path in (source / 'code/train.py', suite / 'code/train.py'):
        path.write_text('# identical frozen source')
    (suite / 'code/tools/run_bp7_e160_to200.py').write_text('# independent driver')
    for frozen in (source, suite):
        entries = [f'{final.sha256_file(path)}  {path.relative_to(frozen / "code")}'
                   for path in sorted((frozen / 'code').rglob('*.py'))]
        (frozen / 'source.sha256').write_text(chr(10).join(entries) + chr(10))
    command = template(data, source)
    # This is the source recipe's real CLI behavior: trainable is a parser default.
    assert '--save' not in command and args_for(command).save == 'trainable'
    if save is not None:
        command += ['--save', save]
    checkpoint = source / 'full_train_e160/training_state.pt'
    checkpoint.write_bytes(b'protected complete e160')
    existing = tmp_path / 'best_e160.zip'
    existing.write_bytes(b'protected submission')
    final.write_json(source / 'status.json', {'stage': 'complete'})
    final.write_json(source / 'launch_manifest.json', dict(data=str(data), official_data=str(official),
                                                        submission_path=str(existing)))
    final.write_json(source / 'commands.json', {'full_train_e160': command})
    manifest = dict(source_suite=str(source), source_epoch=160, data=str(data), official_data=str(official),
                    gpu_memory_fraction=0.35, allocator_config=final.ALLOCATOR,
                    source_checkpoint_sha256=final.sha256_file(checkpoint),
                    existing_e160_submission=str(existing),
                    existing_e160_submission_sha256=final.sha256_file(existing),
                    submission_paths={str(e): str(tmp_path / f'e{e}.zip') for e in (180, 200)})
    state_checks = []
    def verify_state(path, actual_command, epoch, digest):
        assert path == checkpoint and actual_command == command and epoch == 160
        assert digest == final.sha256_file(checkpoint)
        state_checks.append(epoch)
        return {'epoch': epoch}
    monkeypatch.setattr(final, 'verify_state', verify_state)
    monkeypatch.setattr(final, 'validate_zip', lambda path, images: {'sha256': final.sha256_file(path)})
    # Run real frozen-source, manifest, command and output-path input validation.
    if save == 'none':
        with pytest.raises(ValueError, match='trainable inference weights'):
            final.verify_inputs(suite, manifest)
        assert not state_checks
    else:
        actual_command, checked = final.verify_inputs(suite, manifest)
        assert actual_command == command and checked['epoch'] == 160 and state_checks == [160]
        assert checkpoint.read_bytes() == b'protected complete e160'
        assert existing.read_bytes() == b'protected submission'
