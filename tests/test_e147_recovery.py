"""CPU regression tests of direct recovery, progress detection and process cleanup."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_bp7_cos200_final as original
import run_bp7_e147_recovery as recovery
import train


def parse(command):
    return train.parse_args(command[command.index('train.py') + 1:])


def freeze(suite):
    (suite / 'source.sha256').write_text(''.join(
        f'{recovery.sha256_file(p)}  {p.relative_to(suite / "code")}\n'
        for p in sorted((suite / 'code').rglob('*.py'))))


@pytest.fixture
def context(monkeypatch):
    root = Path(__file__).resolve().parents[1] / '.runtime'
    root.mkdir(exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix='t147-', dir=root))
    source, suite, data, official = [base / n for n in ('source', 'new', 'data', 'official')]
    for p in (source / 'code', suite / 'code/tools', source / 'full_train_e160', official / 'val'):
        p.mkdir(parents=True)
    (source / 'code/train.py').write_text('# unchanged frozen train')
    (suite / 'code/train.py').write_text('# unchanged frozen train')
    driver = suite / 'code/tools/run_bp7_e147_recovery.py'
    driver.write_text('# new frozen driver')
    freeze(source)
    freeze(suite)
    (source / 'driver.lock').touch()
    recovery.write_json(source / 'status.json', dict(stage='failed'))
    recovery.write_json(source / 'launch_manifest.json', dict(data=str(data), official_data=str(official)))
    checkpoint = source / 'full_train_e160/training_state.pt'
    checkpoint.write_text('epoch147')
    template = original.training_command('/python', data, source, 'full_train_e160', stop_epoch=160,
                                         resume=source / 'validation/resume_e131/training_state.pt')
    template[template.index('--memory-fraction') + 1] = '0.35'
    recovery.write_json(source / 'commands.json', dict(full_train_e160=template))
    existing, output = base / 'e130.zip', base / 'e160.zip'
    existing.write_bytes(b'protected e130 zip')
    recovery.write_json(official / 'val/images.json', dict(images=[dict(id=i) for i in range(1426)]))
    manifest = dict(source_suite=str(source), source_epoch=147, data=str(data), official_data=str(official),
        source_checkpoint_sha256=recovery.sha256_file(checkpoint), gpu_memory_fraction=0.35,
        allocator_config=recovery.ALLOCATOR, temp_dir=str(base / 't'), submission_path=str(output),
        existing_e130_submission=str(existing), existing_e130_submission_sha256=recovery.sha256_file(existing))
    recovery.write_json(suite / 'launch_manifest.json', manifest)
    monkeypatch.setattr(recovery, '__file__', str(driver))
    monkeypatch.setattr(sys, 'argv', [str(driver), '--suite', str(suite)])
    monkeypatch.setattr(original, 'original_resources', lambda: None)
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
    for key in ('TMPDIR', 'TMP', 'TEMP', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        monkeypatch.setenv(key, 'wrong inherited value')
    calls, checks, stops = [], [], []
    def env_ready():
        assert all(os.environ[k] == manifest['temp_dir'] for k in ('TMPDIR', 'TMP', 'TEMP'))
        assert tempfile.gettempdir() == manifest['temp_dir']
        assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == recovery.ALLOCATOR
        assert 'PYTORCH_ALLOC_CONF' not in os.environ
    def verify_state(path, command, epoch, expected_sha=None):
        env_ready()
        if epoch is None:
            epoch = int(path.read_text().removeprefix('epoch'))
            assert 148 <= epoch <= 160
        assert path.read_text() == f'epoch{epoch}'
        if expected_sha:
            assert recovery.sha256_file(path) == expected_sha
        if epoch == 147:
            assert not calls
        checks.append(epoch)
        return dict(epoch=epoch, global_step=epoch * 587, sha256=recovery.sha256_file(path))
    def stopped(run, command, epoch, images):
        assert epoch == 160 and images == 9392
        assert (run / 'training_state.pt').read_text() == 'epoch160'
        checks.append(epoch)
        return dict(epoch=160, global_step=93920, checkpoint_sha256='verified160', train_hours=15)
    monkeypatch.setattr(recovery, 'verify_state', verify_state)
    monkeypatch.setattr(recovery, 'verify_stopped_run', stopped)
    monkeypatch.setattr(recovery, 'verify_data', lambda *a: (dict(images=[]), dict(count=9392)))
    monkeypatch.setattr(recovery, 'validate_zip', lambda p, images: dict(sha256=recovery.sha256_file(p), path=str(p)))
    monkeypatch.setattr(recovery, 'stop_child', lambda child, scope: stops.append((child.pid, scope)))
    try:
        yield SimpleNamespace(**locals())
    finally:
        shutil.rmtree(base)


@pytest.mark.parametrize('failure', [None, 'resource', 'stall', 'exit', 'late_checkpoint'])
def test_actual_main_direct_resume_and_fail_closed(context, monkeypatch, failure):
    c = context
    first_saved_epoch = 149 if failure == 'late_checkpoint' else 148
    must_fail = failure in ('resource', 'stall', 'exit')
    clock = [0.0]
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: clock[0])
    class Child:
        pid = 424242
        def __init__(self, command, cwd, env, start_new_session, stdout, stderr):
            c.env_ready()
            assert start_new_session and cwd == c.suite / 'code'
            assert all(env[k] == c.manifest['temp_dir'] for k in ('TMPDIR', 'TMP', 'TEMP'))
            assert recovery.value(command, '--memory-fraction') == '0.35'
            assert c.checks[0] == 147
            c.calls.append(command)
            self.command, self.stream, self.count = command, stdout, 0
            self.training = 'train.py' in command
            if self.training:
                args = parse(command)
                assert args.resume == c.checkpoint and args.stop_after_epoch == 160
                assert args.batch_size == 16 and args.workers == 1 and args.save_every == 10
                assert args.epochs == args.schedule_epochs == 200
                assert args.lr == 1e-4 and args.backbone_lr == 3e-4
                assert args.limit is None and args.grad_ckpt and not args.compile
                self.run = args.out / args.name
            else:
                assert c.checks == [147, first_saved_epoch, 160]
                assert env['CUDA_VISIBLE_DEVICES'] == '1'
                assert recovery.value(command, '--weights') == 'ckpt_e160.pt'
                c.output.write_bytes(b'new e160 zip')
        def wait(self, timeout):
            assert timeout == 30
            self.count += 1
            if not self.training:
                return 0
            if failure == 'resource':
                self.stream.write("OSError: [Errno 28] No space left on device: '/tmp/pymp-bad'\n")
                self.stream.flush()
                raise subprocess.TimeoutExpired(self.command, timeout)
            if failure == 'stall':
                clock[0] += 100
                self.stream.write('worker warning without progress\n')
                self.stream.flush()
                raise subprocess.TimeoutExpired(self.command, timeout)
            if failure == 'exit':
                return 1
            epoch = first_saved_epoch if self.count == 1 else 160
            self.stream.write(f'epoch {epoch} iter 587/587 loss 0.1\n')
            self.stream.flush()
            (self.run / 'training_state.pt').write_text(f'epoch{epoch}')
            clock[0] += 30
            if self.count == 1:
                raise subprocess.TimeoutExpired(self.command, timeout)
            return 0
    monkeypatch.setattr(recovery.subprocess, 'Popen', Child)
    if must_fail:
        with pytest.raises((RuntimeError, TimeoutError, subprocess.CalledProcessError)):
            recovery.main()
    else:
        recovery.main()
    status = recovery.read_json(c.suite / 'status.json')
    assert c.checkpoint.read_text() == 'epoch147' and c.existing.read_bytes() == b'protected e130 zip'
    assert 'child_pid' not in status
    assert len([cmd for cmd in c.calls if 'train.py' in cmd]) == 1
    if must_fail:
        assert status['stage'] == 'failed' and len(c.calls) == 1
        assert not c.output.exists() and c.stops == [(424242, c.suite)]
    else:
        assert status['stage'] == 'complete' and len(c.calls) == 2 and not c.stops
        assert c.checks == [147, first_saved_epoch, 160]
        assert recovery.read_json(c.suite / 'checkpoint_after_resume_validation.json')['global_step'] == first_saved_epoch * 587
        assert c.output.with_suffix('.sha256').exists()


def test_only_increasing_iterations_reset_watchdog():
    progress, last = recovery.check_progress('epoch 148 iter 50/587', (147, 587), 0, 30, True)
    assert progress == (148, 50) and last == 30
    progress, last = recovery.check_progress('epoch 148 iter 50/587\nwarning', progress, last, 200, True)
    assert last == 30
    with pytest.raises(TimeoutError):
        recovery.check_progress('epoch 148 iter 50/587\nmore warnings', progress, last, 330, True)
    assert recovery.check_progress('', progress, last, 4000, False)[0] == progress


def test_cleanup_targets_only_this_suite_groups(monkeypatch, tmp_path):
    suite = tmp_path / 'suite'
    own = ['python', 'train.py', '--out', str(suite)]
    foreign = ['llama-server', '--model', 'other']
    table = {100: dict(group=100, start='1', argv=own),
             200: dict(group=200, start='2', argv=own),
             201: dict(group=200, start='3', argv=own),
             300: dict(group=300, start='4', argv=foreign),
             400: dict(group=400, start='5', argv=own),
             401: dict(group=400, start='6', argv=foreign)}
    groups, pids = [], []
    monkeypatch.setattr(recovery, 'process_table', lambda: table)
    monkeypatch.setattr(recovery.os, 'getpgrp', lambda: 999)
    monkeypatch.setattr(recovery.os, 'killpg', lambda group, sig: groups.append((group, sig)))
    monkeypatch.setattr(recovery.os, 'kill', lambda pid, sig: pids.append((pid, sig)))
    recovery.kill_remaining_owned(suite)
    assert groups == [(100, signal.SIGKILL), (200, signal.SIGKILL)]
    assert pids == [(400, signal.SIGKILL)]
    assert not recovery.owned_command(['python', '--suite', str(suite)], suite)


def test_stop_launcher_then_clean_orphan_rank_groups(monkeypatch, tmp_path):
    events = []
    class Child:
        def poll(self): return None
        def terminate(self): events.append('TERM launcher')
        def wait(self, timeout):
            assert timeout == 10
            events.append('wait')
            if events.count('wait') == 1:
                raise subprocess.TimeoutExpired('launcher', timeout)
            return -9
    monkeypatch.setattr(recovery, 'kill_remaining_owned', lambda suite: events.append('KILL owned ranks'))
    monkeypatch.setattr(recovery, 'process_table', lambda: {})
    recovery.stop_child(Child(), tmp_path)
    assert events == ['TERM launcher', 'wait', 'KILL owned ranks', 'wait']


@pytest.mark.parametrize('fault', [None, 'epoch', 'scheduler', 'optimizer', 'rng', 'args', 'lr', 'tensor', 'sha'])
def test_complete_checkpoint_validation(context, monkeypatch, fault):
    # Restore the real validator for a tiny synthetic model, no CUDA allocation.
    import torch
    import importlib.util
    specification = importlib.util.spec_from_file_location('bp7_state_test',
        Path(__file__).resolve().parents[1] / 'tools/run_bp7_e147_recovery.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    c = context
    args = parse(c.template)
    config = json.loads(json.dumps({k: v for k, v in vars(args).items() if k not in recovery.IGNORED_ARGS}, default=str))
    manifests = {}
    for split in ('train', 'val', 'test'):
        for rel in (f'classification/{split}.csv', f'detection/instances_{split}.json'):
            path = c.data / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('data')
            manifests[rel] = recovery.sha256_file(path)
    factor = train.lr_factor(86289, 587, 200 * 587)
    state = dict(format_version=1, epoch=147, global_step=86289, scheduler={'last_epoch': 86289},
        model={'p': torch.ones(2)}, ema=None, history=[], train_hours=14,
        optimizer=dict(state={0: {'step': torch.tensor(86289)}},
                       param_groups=[dict(initial_lr=1e-4, lr=1e-4 * factor)]),
        rank_rng=[dict(python=(), numpy=(), torch=torch.tensor([1]), cuda=torch.tensor([2])) for _ in range(2)],
        signature=dict(args=config, world_size=2, dataset_size=9392, steps_per_epoch=587,
                       manifests=manifests, parameters={'p': dict(shape=[2], trainable=True)}))
    if fault == 'epoch': state['epoch'] = 146
    elif fault == 'scheduler': state['scheduler']['last_epoch'] -= 1
    elif fault == 'optimizer': state['optimizer']['state'][0]['step'] -= 1
    elif fault == 'rng': state['rank_rng'][1]['cuda'] = None
    elif fault == 'args': state['signature']['args']['workers'] = 2
    elif fault == 'lr': state['optimizer']['param_groups'][0]['lr'] = 1e-4
    elif fault == 'tensor': state['model']['p'][0] = float('nan')
    checkpoint = c.base / 'synthetic.pt'
    torch.save(state, checkpoint)
    digest = 'wrong hash' if fault == 'sha' else recovery.sha256_file(checkpoint)
    if fault:
        with pytest.raises(ValueError):
            module.verify_state(checkpoint, c.template, 147, digest, budget=(2, 2))
    else:
        assert module.verify_state(checkpoint, c.template, 147, digest, budget=(2, 2))['global_step'] == 86289
