"""CPU regression checks for the cos200 controller and GPU wrapper."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_bp7_cos200_final as final
import train_gpu_limited as limited
import train


def parse(command):
    return train.parse_args(command[command.index('train.py') + 1:])


def test_recipe_and_exact_resume_signature(tmp_path):
    import torch
    from sarbench.training_state import compatibility_signature, validate_compatibility
    data = tmp_path / 'data'
    for split in ('train', 'val', 'test'):
        for rel in (f'classification/{split}.csv', f'detection/instances_{split}.json'):
            p = data / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('same synthetic manifest')
    class Loader:
        dataset, sampler = range(9392), object()
        def __len__(self): return 587
    first = final.training_command('/python', data, tmp_path)
    path = tmp_path / 'full_train/training_state.pt'
    second = final.training_command('/python', data, tmp_path, 'full_train_e160', stop_epoch=160, resume=path)
    a, b = parse(first), parse(second)
    assert a.epochs == a.schedule_epochs == b.epochs == b.schedule_epochs == 200
    assert a.stop_after_epoch == 130 and b.stop_after_epoch == 160 and b.resume == path
    assert a.name == 'full_train' and b.name == 'full_train_e160'
    assert a.lr == b.lr == 1e-4 and a.backbone_lr == b.backbone_lr == 3e-4
    assert a.batch_size == 16 and a.workers == b.workers == 1
    assert a.grad_ckpt and not a.compile and a.reproducible_loader and a.save_training_state
    assert a.train_split == 'train+val+test' and a.eval_splits == ''
    i = first.index('tools/train_gpu_limited.py')
    assert first[i:i+4] == ['tools/train_gpu_limited.py', '--memory-fraction', '0.2', 'train.py']
    model = torch.nn.Linear(2, 1)
    saved = compatibility_signature(a, model, Loader(), 2)
    validate_compatibility(saved, compatibility_signature(b, model, Loader(), 2))
    b.workers = 2
    with pytest.raises(ValueError, match='workers'):
        validate_compatibility(saved, compatibility_signature(b, model, Loader(), 2))


@pytest.fixture
def stopped(tmp_path):
    import torch
    def make(epoch=130):
        run = tmp_path / f'epoch{epoch}'
        run.mkdir()
        command = final.training_command('/python', tmp_path / 'data', tmp_path, run.name, stop_epoch=epoch)
        args = json.loads(json.dumps(vars(parse(command)), default=str))
        (run / 'log.txt').write_text(json.dumps(args) + '\nparameters: 4 total, 2 trainable (+ 1 floating-point buffer values)\n'
            + f'Stopped cleanly after epoch {epoch}; resume from training_state.pt\n')
        ignored = {'out', 'name', 'resume', 'stop_after_epoch', 'save_training_state', 'save', 'save_every', 'dump_raw', 'allow_over_budget'}
        steps = epoch * 587
        factor = train.lr_factor(steps, 587, 200 * 587)
        state = dict(format_version=1, epoch=epoch, global_step=steps, scheduler=dict(last_epoch=steps),
            history=[], ema=None, rank_rng=[{}, {}], train_hours=1.0,
            signature=dict(world_size=2, dataset_size=9392, steps_per_epoch=587,
                args={k: v for k, v in args.items() if k not in ignored}, manifests={},
                parameters={'frozen': dict(shape=[2], trainable=False), 'trained': dict(shape=[2], trainable=True)}),
            model=dict(frozen=torch.zeros(2), trained=torch.tensor([1.0, 2.0]), persistent=torch.ones(1)),
            optimizer=dict(state={0: dict(step=torch.tensor(float(steps)))},
                param_groups=[dict(initial_lr=1e-4, lr=1e-4 * factor)]))
        torch.save(state, run / 'training_state.pt')
        torch.save(dict(trained=state['model']['trained'], persistent=torch.ones(1)), run / f'ckpt_e{epoch}.pt')
        return run, command, state
    return make


@pytest.mark.parametrize('epoch,updates', [(130, 76310), (160, 93920)])
def test_correct_cos200_stops(stopped, epoch, updates):
    run, command, _ = stopped(epoch)
    result = final.verify_stopped_run(run, command, epoch, 9392, expected_budget=(4, 2, 1))
    assert result['global_step'] == updates and result['schedule_epochs'] == 200
    assert result['optimizer_and_rank_rng_saved'] and result['trained_tensors_match']


@pytest.mark.parametrize('fault', ['old_160_lr', 'wrong_update', 'missing_rank', 'changed_tensor', 'wrong_schedule'])
def test_reject_wrong_state(stopped, fault):
    import torch
    run, command, state = stopped()
    if fault == 'old_160_lr': state['optimizer']['param_groups'][0]['lr'] = 1e-4 * train.lr_factor(76310, 587, 160 * 587)
    elif fault == 'wrong_update': state['global_step'] -= 1
    elif fault == 'missing_rank': state['rank_rng'].pop()
    elif fault == 'changed_tensor': state['model']['trained'][0] = 10.0
    elif fault == 'wrong_schedule': state['signature']['args']['schedule_epochs'] = 160
    torch.save(state, run / 'training_state.pt')
    with pytest.raises(ValueError): final.verify_stopped_run(run, command, 130, 9392, expected_budget=(4, 2, 1))


@pytest.mark.parametrize('fraction', ['0', '-0.1', '0.21', '1'])
def test_reject_excess_memory(fraction):
    with pytest.raises(SystemExit): limited.parse_args(['--memory-fraction', fraction, 'train.py'])


@pytest.mark.parametrize('rank,script', [('1', 'train.py'), (None, 'predict.py')])
def test_limit_correct_device_before_run(monkeypatch, rank, script):
    calls = []
    if rank is None: monkeypatch.delenv('LOCAL_RANK', raising=False)
    else: monkeypatch.setenv('LOCAL_RANK', rank)
    cuda = SimpleNamespace(device_count=lambda: 2, set_device=lambda d: calls.append(('device', d)),
        set_per_process_memory_fraction=lambda f, device: calls.append(('limit', f, device)),
        max_memory_allocated=lambda d: 12, max_memory_reserved=lambda d: 20)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(limited.runpy, 'run_path', lambda path, run_name: calls.append(('run', Path(path).name, list(sys.argv))))
    monkeypatch.setattr(sys, 'argv', ['train_gpu_limited.py'])
    limited.main(['--memory-fraction', '0.2', script, '--workers', '1'])
    assert calls[:2] == [('device', int(rank or 0)), ('limit', 0.2, int(rank or 0))]
    assert calls[2][:2] == ('run', script) and calls[2][2][1:] == ['--workers', '1']



def test_recovery_allocator_and_probe_keep_training_recipe(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(final, 'original_resources', lambda: None)
    monkeypatch.setenv('PYTORCH_ALLOC_CONF', 'max_split_size_mb:64')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '2')
    final.configure_resources('expandable_segments:True')
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
    assert 'PYTORCH_ALLOC_CONF' not in os.environ
    assert os.environ['OMP_NUM_THREADS'] == '1'
    command = final.recovery_probe_command('/python', tmp_path / 'data', tmp_path / 'validation', tmp_path / 'source.pt')
    args = parse(command)
    assert args.stop_after_epoch == 2 and args.epochs == args.schedule_epochs == 200
    assert args.batch_size == 16 and args.workers == 1 and args.limit is None
    assert args.save_every == 1 and args.resume == tmp_path / 'source.pt'
    assert args.grad_ckpt and not args.compile and args.reproducible_loader
    assert args.lr == 1e-4 and args.backbone_lr == 3e-4


@pytest.fixture
def recovery(stopped, tmp_path):
    run, _, state = stopped(1)
    source = tmp_path / 'failed_source'
    source.mkdir()
    (source / 'driver.lock').touch()
    (source / 'status.json').write_text(json.dumps(dict(stage='failed')))
    (source / 'launch_manifest.json').write_text(json.dumps(dict(data_sha256={})))
    target = source / 'full_train'
    run.rename(target)
    checkpoint = target / 'training_state.pt'
    manifest = dict(data_sha256={}, recovery=dict(source_suite=str(source), checkpoint=str(checkpoint),
        checkpoint_sha256=final.sha256_file(checkpoint), epoch=1, global_step=587,
        allocator_config='expandable_segments:True'))
    return manifest, tmp_path / 'data', state


def test_recovery_verifies_saved_e1_state(recovery):
    manifest, data, _ = recovery
    result = final.verify_recovery(manifest, data, expected_budget=(4, 2, 1))
    assert result['epoch'] == 1 and result['global_step'] == 587
    assert result['optimizer_and_rank_rng_saved']
    assert result['allocator_config'] == 'expandable_segments:True'


@pytest.mark.parametrize('fault', ['running', 'hash', 'rank_rng', 'steps', 'workers', 'lr', 'outside'])
def test_recovery_rejects_wrong_source(recovery, fault):
    import torch
    manifest, data, state = recovery
    record = manifest['recovery']
    source, checkpoint = Path(record['source_suite']), Path(record['checkpoint'])
    if fault == 'running':
        (source / 'status.json').write_text(json.dumps(dict(stage='training')))
    elif fault == 'hash':
        record['checkpoint_sha256'] = 'wrong digest'
    elif fault == 'outside':
        record['checkpoint'] = str(source / 'another.pt')
    else:
        if fault == 'rank_rng': state['rank_rng'].pop()
        elif fault == 'steps': state['global_step'] = 588
        elif fault == 'workers': state['signature']['args']['workers'] = 2
        elif fault == 'lr': state['optimizer']['param_groups'][0]['lr'] = 9e-5
        torch.save(state, checkpoint)
        record['checkpoint_sha256'] = final.sha256_file(checkpoint)
    with pytest.raises(ValueError):
        final.verify_recovery(manifest, data, expected_budget=(4, 2, 1))



@pytest.mark.parametrize('probe_exit_code', [0, 1])
def test_main_recovery_executes_resume_chain(tmp_path, monkeypatch, probe_exit_code):
    """Run the actual controller main; mock only costly IO/GPU subprocess work."""
    import os
    import subprocess
    suite = tmp_path / 'recovered'
    (suite / 'code/tools').mkdir(parents=True)
    official = tmp_path / 'official'
    (official / 'val').mkdir(parents=True)
    images = dict(images=[dict(id=i, width=32, height=32) for i in range(9)])
    (official / 'val/images.json').write_text(json.dumps(images))
    source = tmp_path / 'source/full_train/training_state.pt'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'original e1 state')
    manifest = dict(data=str(tmp_path / 'data'), official_data=str(official),
        dependency_suite=str(tmp_path / 'dependency'), submission_paths={
            '130': str(tmp_path / 'e130.zip'), '160': str(tmp_path / 'e160.zip')},
        recovery=dict(source_suite=str(source.parent.parent), checkpoint=str(source),
            checkpoint_sha256='original digest', epoch=1, global_step=587,
            allocator_config='expandable_segments:True'))
    (suite / 'launch_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(final, '__file__', str(suite / 'code/tools/run_bp7_cos200_final.py'))
    monkeypatch.setattr(sys, 'argv', ['run_bp7_cos200_final.py', '--suite', str(suite)])
    monkeypatch.setattr(final, 'original_resources', lambda: None)
    monkeypatch.setenv('PYTORCH_ALLOC_CONF', 'wrong inherited alias')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'wrong inherited value')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '2')
    events, training, predictions, validations = [], [], [], []
    monkeypatch.setattr(final, 'verify_snapshot', lambda *args: dict(source_files=3))
    monkeypatch.setattr(final, 'verify_data', lambda *args: (images, dict(count=9392)))

    def recovered(record, data):
        assert not training
        assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
        assert 'PYTORCH_ALLOC_CONF' not in os.environ
        events.append('recovery_verified')
        assert record['recovery']['checkpoint'] == str(source)
        return dict(checkpoint=str(source), epoch=1, global_step=587,
                    allocator_config='expandable_segments:True')
    monkeypatch.setattr(final, 'verify_recovery', recovered)

    class Child:
        def __init__(self, command, cwd, env, stdout, stderr):
            assert events and events[0] == 'recovery_verified'
            assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
            assert Path(cwd) == suite / 'code'
            assert command[command.index('--memory-fraction') + 1] == '0.2'
            self.pid = 9000 + len(training) + len(predictions)
            self.returncode = 0
            if 'train.py' in command:
                args = parse(command)
                assert args.resume is not None and args.resume.is_file()
                assert args.limit is None and args.workers == 1 and args.batch_size == 16
                assert args.epochs == args.schedule_epochs == 200
                assert args.lr == 1e-4 and args.backbone_lr == 3e-4
                training.append(args)
                events.append(f'train{args.stop_after_epoch}')
                run = args.out / args.name
                run.mkdir(parents=True, exist_ok=True)
                (run / 'training_state.pt').write_text(f'epoch {args.stop_after_epoch}')
                (run / f'ckpt_e{args.stop_after_epoch}.pt').write_text('mock inference checkpoint')
                if args.stop_after_epoch == 2:
                    self.returncode = probe_exit_code
            else:
                assert 'predict.py' in command
                assert env['CUDA_VISIBLE_DEVICES'] == '1'
                assert env['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
                predictions.append(list(command))
                Path(command[command.index('--out') + 1]).write_bytes(b'validated by mocked scorer')
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def wait(self, timeout): return self.returncode
    monkeypatch.setattr(final.subprocess, 'Popen', Child)

    def stopped_check(run, command, epoch, count):
        assert count == 9392
        assert (run / 'training_state.pt').read_text() == f'epoch {epoch}'
        validations.append(epoch)
        return dict(epoch=epoch, schedule_epochs=200, global_step=587 * epoch,
                    checkpoint_sha256=f'checkpoint {epoch}', train_hours=epoch / 20)
    monkeypatch.setattr(final, 'verify_stopped_run', stopped_check)
    monkeypatch.setattr(final, 'validate_zip', lambda output, image_list: dict(
        sha256='mock validated digest', prediction_images=len(json.loads(Path(image_list).read_text())['images']),
        format_passed=True, path=str(output)))

    if probe_exit_code:
        with pytest.raises(subprocess.CalledProcessError):
            final.main()
        assert [a.stop_after_epoch for a in training] == [2]
        assert not predictions and not validations
        assert json.loads((suite / 'status.json').read_text())['stage'] == 'failed'
        assert not (suite / 'full_train').exists()
    else:
        final.main()
        assert [a.stop_after_epoch for a in training] == [2, 130, 160]
        assert [a.resume for a in training] == [source,
            suite / 'validation/resume_probe/training_state.pt', suite / 'full_train/training_state.pt']
        assert training[0].save_every == 1 and training[0].name == 'resume_probe'
        assert validations == [2, 130, 160]
        assert len(predictions) == 3
        assert predictions[0][predictions[0].index('--weights') + 1] == 'ckpt_e2.pt'
        assert not (suite / 'validation/smoke').exists()
        result = json.loads((suite / 'status.json').read_text())
        assert result['stage'] == 'complete' and set(result['submissions']) == {'130', '160'}
        assert result['allocator_config'] == 'expandable_segments:True'
        assert (suite / 'full_train/training_state.pt').read_text() == 'epoch 130'
    assert source.read_bytes() == b'original e1 state'
