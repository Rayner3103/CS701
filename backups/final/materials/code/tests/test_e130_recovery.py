"""CPU checks of the actual e130 recovery main, with simulated GPU children."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_bp7_cos200_final as original
import run_bp7_e130_recovery as recovery
import train


def parse(command):
    return train.parse_args(command[command.index('train.py') + 1:])


def freeze(suite):
    entries = [f'{recovery.sha256_file(p)}  {p.relative_to(suite / "code")}'
               for p in sorted((suite / 'code').rglob('*.py'))]
    (suite / 'source.sha256').write_text('\n'.join(entries) + '\n')


@pytest.fixture
def setup_main(tmp_path, monkeypatch):
    source, suite, data, official = [tmp_path / name for name in ('source', 'recovery', 'data', 'official')]
    for path in (source / 'code/tools', suite / 'code/tools', source / 'full_train', official / 'val'):
        path.mkdir(parents=True)
    required = ['train.py', 'predict.py', 'score_submission.py', 'tools/train_gpu_limited.py',
                'tools/run_bp7_cos200_final.py', 'tools/run_bp7_full_stop130.py',
                'tools/run_bp7_schedule_search.py', 'tools/run_bp7_lr_trials.py']
    for name in required:
        text = '# original frozen implementation\n'
        if name.endswith('train_gpu_limited.py'):
            text += '# cap 0.20 and error 0.20\n'
        (source / 'code' / name).write_text(text)
        (suite / 'code' / name).write_text(text.replace('0.20', '0.35'))
    driver = suite / 'code/tools/run_bp7_e130_recovery.py'
    driver.write_text('# newly frozen recovery driver\n')
    freeze(source)
    freeze(suite)
    (source / 'driver.lock').touch()
    recovery.write_json(source / 'status.json', {'stage': 'failed'})
    checkpoint = source / 'full_train/training_state.pt'
    checkpoint.write_text('epoch 130')
    (source / 'full_train/ckpt_e130.pt').write_text('weights 130')
    manifests = {}
    for split in ('train', 'val', 'test'):
        for name in (f'classification/{split}.csv', f'detection/instances_{split}.json'):
            path = data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('unchanged data')
            manifests[name] = recovery.sha256_file(path)
    images = dict(images=[dict(id=i, width=32, height=32) for i in range(1426)])
    recovery.write_json(official / 'val/images.json', images)
    recovery.write_json(source / 'launch_manifest.json', dict(data=str(data), official_data=str(official),
                                                            data_sha256=manifests))
    commands = dict(
        full_train=original.training_command('/python', data, source, stop_epoch=130,
                    resume=source / 'validation/resume_probe/training_state.pt'),
        full_train_e160=original.training_command('/python', data, source, 'full_train_e160',
                    stop_epoch=160, resume=checkpoint))
    recovery.write_json(source / 'commands.json', commands)
    existing, output = tmp_path / 'e130.zip', tmp_path / 'e160.zip'
    existing.write_bytes(b'original e130 submission')
    manifest = dict(source_suite=str(source), data=str(data), official_data=str(official),
                    gpu_memory_fraction=0.35, allocator_config='expandable_segments:True',
                    source_checkpoint_sha256=recovery.sha256_file(checkpoint), submission_path=str(output),
                    existing_e130_submission=str(existing), existing_e130_submission_sha256=recovery.sha256_file(existing))
    recovery.write_json(suite / 'launch_manifest.json', manifest)
    monkeypatch.setattr(recovery, '__file__', str(driver))
    monkeypatch.setattr(sys, 'argv', [str(driver), '--suite', str(suite)])
    # Keep real allocator/thread setup while avoiding pytest affinity/nice changes.
    monkeypatch.setattr(original, 'original_resources', lambda: None)
    monkeypatch.setenv('PYTORCH_ALLOC_CONF', 'wrong inherited alias')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'wrong inherited value')
    events, children, validated = [], [], []
    def allocator_ready():
        assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
        assert 'PYTORCH_ALLOC_CONF' not in os.environ
        assert os.environ['OMP_NUM_THREADS'] == '1'
    def check_stopped(run, command, epoch, count):
        allocator_ready()
        assert count == 9392
        assert (run / 'training_state.pt').read_text() == f'epoch {epoch}'
        args = parse(command)
        assert args.stop_after_epoch == epoch and args.epochs == args.schedule_epochs == 200
        if epoch == 130:
            assert not children
            assert command == commands['full_train']
        events.append(('verify', epoch))
        validated.append(epoch)
        return dict(epoch=epoch, global_step=epoch * 587, checkpoint_sha256=f'weights {epoch}', train_hours=12)
    def check_zip(path, image_list):
        allocator_ready()
        assert path.exists()
        events.append(('zip', str(path)))
        return dict(path=str(path), sha256=recovery.sha256_file(path), format_passed=True,
                    prediction_images=len(recovery.read_json(image_list)['images']))
    def check_data(data_path, official_path):
        assert data_path == data and official_path == official
        allocator_ready()
        return images, dict(count=9392, matches_official_training=True)
    monkeypatch.setattr(recovery, 'verify_stopped_run', check_stopped)
    monkeypatch.setattr(recovery, 'validate_zip', check_zip)
    monkeypatch.setattr(recovery, 'verify_data', check_data)
    return dict(source=source, suite=suite, data=data, official=official, checkpoint=checkpoint,
                existing=existing, output=output, manifest=manifest, commands=commands,
                events=events, children=children, validated=validated, allocator_ready=allocator_ready)


@pytest.mark.parametrize('fail_stage', [None, 'probe_training', 'probe_prediction'])
def test_actual_main_resume_chain_and_fail_closed(setup_main, monkeypatch, fail_stage):
    ctx, waits = setup_main, []
    suite = ctx['suite']
    class Child:
        pid = 24680
        def __init__(self, command, cwd, env, stdout, stderr):
            ctx['allocator_ready']()
            assert cwd == suite / 'code'
            assert recovery.value(command, '--memory-fraction') == '0.35'
            assert 130 in ctx['validated']
            assert ('zip', str(ctx['existing'])) in ctx['events']
            ctx['children'].append(list(command))
            self.command, self.waited, self.returncode = command, False, 0
            if 'train.py' in command:
                args = parse(command)
                assert args.batch_size == 16 and args.workers == 1
                assert args.lr == 1e-4 and args.backbone_lr == 3e-4
                assert args.epochs == args.schedule_epochs == 200
                assert args.limit is None and args.grad_ckpt and not args.compile
                expected_epoch = 130 if args.stop_after_epoch == 131 else 131
                assert args.resume.read_text() == f'epoch {expected_epoch}'
                if args.stop_after_epoch == 160:
                    assert 131 in ctx['validated']
                    assert (suite / 'validation/probe_passed.json').exists()
                run = args.out / args.name
                (run / 'training_state.pt').write_text(f'epoch {args.stop_after_epoch}')
                (run / f'ckpt_e{args.stop_after_epoch}.pt').write_text('mock weights')
                if fail_stage == 'probe_training' and args.stop_after_epoch == 131:
                    self.returncode = 1
            else:
                assert 'predict.py' in command and env['CUDA_VISIBLE_DEVICES'] == '1'
                assert env['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True'
                assert 'PYTORCH_ALLOC_CONF' not in env
                for flag, expected in (('--batch-size', '2'), ('--workers', '0'), ('--shift', '-0.2')):
                    assert recovery.value(command, flag) == expected
                epoch = 131 if recovery.value(command, '--weights') == 'ckpt_e131.pt' else 160
                assert epoch in ctx['validated']
                output = Path(recovery.value(command, '--out'))
                assert not output.exists()
                output.write_bytes(b'mock predictions')
                if fail_stage == 'probe_prediction' and epoch == 131:
                    self.returncode = 2
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def wait(self, timeout):
            assert timeout == 30
            waits.append(timeout)
            if not self.waited:
                self.waited = True
                raise subprocess.TimeoutExpired(self.command, timeout)
            return self.returncode
    monkeypatch.setattr(recovery.subprocess, 'Popen', Child)
    if fail_stage:
        with pytest.raises(subprocess.CalledProcessError):
            recovery.main()
    else:
        recovery.main()
    training = [parse(c) for c in ctx['children'] if 'train.py' in c]
    predictions = [c for c in ctx['children'] if 'predict.py' in c]
    status = recovery.read_json(suite / 'status.json')
    assert ctx['checkpoint'].read_text() == 'epoch 130'
    assert ctx['existing'].read_bytes() == b'original e130 submission'
    assert status['allocator_config'] == 'expandable_segments:True'
    assert status['gpu_memory_fraction'] == 0.35 and 'child_pid' not in status
    assert training[0].resume == ctx['checkpoint'] and training[0].save_every == 1
    assert training[0].name == 'resume_e131'
    assert recovery.read_json(suite / 'commands.json')
    if fail_stage:
        assert status['stage'] == 'failed'
        assert [a.stop_after_epoch for a in training] == [131]
        assert not (suite / 'full_train_e160').exists() and not ctx['output'].exists()
        assert len(predictions) == (fail_stage == 'probe_prediction')
    else:
        assert [a.stop_after_epoch for a in training] == [131, 160]
        assert training[1].resume == suite / 'validation/resume_e131/training_state.pt'
        assert training[1].save_every == 10 and ctx['validated'] == [130, 131, 160]
        assert len(predictions) == 2 and len(waits) == 8
        assert status['stage'] == 'complete' and set(status['submissions']) == {'130', '160'}
        assert recovery.read_json(suite / 'checkpoint_e131_validation.json')['global_step'] == 76897
        assert recovery.read_json(suite / 'checkpoint_e160_validation.json')['global_step'] == 93920
        assert recovery.read_json(suite / 'validation/probe_passed.json')['submission']['prediction_images'] == 8
        assert recovery.read_json(suite / 'submission_e160_validation.json')['prediction_images'] == 1426
        assert ctx['output'].with_suffix('.sha256').exists()
        assert (suite / 'validation/resume_e131/training_state.pt').read_text() == 'epoch 131'


@pytest.mark.parametrize('fault', ['source_active', 'source_hash', 'new_hash', 'checkpoint_hash',
                                  'submission_hash', 'output_exists', 'recipe_change', 'wrapper_change'])
def test_actual_main_rejects_invalid_inputs_before_any_child(setup_main, monkeypatch, fault):
    c = setup_main
    source, suite = c['source'], c['suite']
    if fault == 'source_active':
        recovery.write_json(source / 'status.json', {'stage': 'training'})
    elif fault in ('source_hash', 'new_hash'):
        root = source if fault == 'source_hash' else suite
        (root / 'code/train.py').write_text('changed')
    elif fault == 'checkpoint_hash':
        c['checkpoint'].write_text('changed state')
    elif fault == 'submission_hash':
        c['existing'].write_bytes(b'changed zip')
    elif fault == 'output_exists':
        c['output'].write_bytes(b'protected existing e160')
    elif fault == 'recipe_change':
        command = c['commands']['full_train_e160']
        command[command.index('--workers') + 1] = '2'
        recovery.write_json(source / 'commands.json', c['commands'])
    elif fault == 'wrapper_change':
        (suite / 'code/tools/train_gpu_limited.py').write_text('# unlimited allocator')
        freeze(suite)
    monkeypatch.setattr(recovery.subprocess, 'Popen', lambda *a, **k: pytest.fail('Invalid inputs launched a child'))
    with pytest.raises((ValueError, FileExistsError)):
        recovery.main()
    assert recovery.read_json(suite / 'status.json')['stage'] == 'failed'
    assert not (suite / 'validation').exists()


def test_source_lock_prevents_concurrent_resume(setup_main, monkeypatch):
    c = setup_main
    monkeypatch.setattr(recovery.subprocess, 'Popen', lambda *a, **k: pytest.fail('Locked source launched child'))
    with (c['source'] / 'driver.lock').open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            recovery.main()
    assert recovery.read_json(c['suite'] / 'status.json')['stage'] == 'failed'


def test_continuation_keeps_exact_training_signature(setup_main):
    import torch
    from sarbench.training_state import compatibility_signature, validate_compatibility
    c = setup_main
    template = c['commands']['full_train_e160']
    probe = recovery.continuation_command(template, c['suite'] / 'validation', 'resume_e131',
                                          131, c['checkpoint'], 1)
    final = recovery.continuation_command(template, c['suite'], 'full_train_e160', 160,
                                          c['suite'] / 'validation/resume_e131/training_state.pt', 10)
    class Loader:
        dataset, sampler = range(9392), object()
        def __len__(self): return 587
    model = torch.nn.Linear(2, 1)
    original_signature = compatibility_signature(parse(c['commands']['full_train']), model, Loader(), 2)
    for command in (probe, final):
        validate_compatibility(original_signature, compatibility_signature(parse(command), model, Loader(), 2))
        for index, (before, after) in enumerate(zip(template, command)):
            if before != after:
                assert template[index - 1] in recovery.MUTABLE_FLAGS
    assert recovery.value(template, '--memory-fraction') == '0.2'
