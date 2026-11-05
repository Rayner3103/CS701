"""One BP7 full-data trajectory: cosine 200, export e130 then continue to e160."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from run_bp7_full_stop130 import training_command as original_command, validate_zip
from run_bp7_schedule_search import configure_resources as original_resources
from run_bp7_schedule_search import read_json, sha256_file, stamp, write_json

EXPECTED_BUDGET = (125076960, 39867360, 16)
MEMORY_FRACTION = 0.20


def training_command(python, data, suite, name='full_train', *, stop_epoch=130,
                     resume=None, memory_fraction=MEMORY_FRACTION):
    if stop_epoch not in (1, 2, 130, 160):
        raise ValueError('Only the smoke, recovery probe, e130 and e160 boundaries are authorized')
    if memory_fraction != MEMORY_FRACTION:
        raise ValueError('Unexpected concurrent GPU memory budget')
    command = original_command(python, data, suite, name)
    for flag, value in {'--epochs': '200', '--schedule-epochs': '200',
                        '--stop-after-epoch': str(stop_epoch), '--workers': '1'}.items():
        command[command.index(flag) + 1] = value
    command.remove('--compile')
    position = command.index('train.py')
    command[position:position] = ['tools/train_gpu_limited.py', '--memory-fraction', str(memory_fraction)]
    if resume is not None:
        command += ['--resume', str(resume)]
    return command


def configure_resources(allocator_config=None):
    original_resources()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if allocator_config is not None:
        if allocator_config != 'expandable_segments:True':
            raise ValueError('Unexpected recovery CUDA allocator configuration')
        # Set before the controller or any subprocess imports torch. The newer
        # alias must not override the explicit compatibility setting below.
        os.environ.pop('PYTORCH_ALLOC_CONF', None)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = allocator_config


def recovery_probe_command(python, data, validation, checkpoint):
    command = training_command(python, data, validation, 'resume_probe',
                               stop_epoch=2, resume=checkpoint)
    command[command.index('--save-every') + 1] = '1'
    return command


def verify_recovery(manifest, data, *, expected_budget=EXPECTED_BUDGET):
    """Accept only the authentic full-data e1 state of the failed concurrent run."""
    recovery = manifest['recovery']
    source = Path(recovery['source_suite']).resolve()
    checkpoint = Path(recovery['checkpoint']).resolve()
    if recovery['allocator_config'] != 'expandable_segments:True':
        raise ValueError('Unexpected recovery allocator configuration')
    if recovery['epoch'] != 1 or recovery['global_step'] != 587:
        raise ValueError('Only the saved e1/587-update recovery point is authorized')
    if checkpoint != (source / 'full_train/training_state.pt').resolve():
        raise ValueError('Recovery checkpoint is outside the original full training run')
    with (source / 'driver.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read_json(source / 'status.json')['stage'] != 'failed':
            raise ValueError('Recovery source must be failed and no longer running')
        original = read_json(source / 'launch_manifest.json')
        if original['data_sha256'] != manifest['data_sha256']:
            raise ValueError('Recovery data manifests differ from the failed run')
        if sha256_file(checkpoint) != recovery['checkpoint_sha256']:
            raise ValueError('Recovery checkpoint SHA256 mismatch')
        import torch
        import train
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    signature = state['signature']
    if (state.get('format_version') != 1 or state['epoch'] != 1
            or state['global_step'] != 587 or state['scheduler']['last_epoch'] != 587
            or signature['world_size'] != 2 or signature['dataset_size'] != 9392
            or signature['steps_per_epoch'] != 587 or len(state['rank_rng']) != 2
            or state['ema'] is not None or state['history']):
        raise ValueError('Invalid e1 recovery progress, coverage or rank RNG state')
    command = training_command(sys.executable, data, source)
    expected = json.loads(json.dumps(vars(train.parse_args(command[command.index('train.py') + 1:])), default=str))
    for key, value in signature['args'].items():
        if key not in expected or expected[key] != value:
            raise ValueError(f'Recovery configuration mismatch: {key}')
    if signature['manifests'] != manifest['data_sha256']:
        raise ValueError('Checkpoint training data digests differ from the frozen manifest')
    for name, digest in signature['manifests'].items():
        if sha256_file(Path(data) / name) != digest:
            raise ValueError(f'Recovery training manifest changed: {name}')
    parameters = signature['parameters']
    total = sum(math.prod(p['shape']) for p in parameters.values())
    trained = sum(math.prod(p['shape']) for p in parameters.values() if p['trainable'])
    if (total, trained) != tuple(expected_budget[:2]):
        raise ValueError('Recovery model parameter budget mismatch')
    for name, specification in parameters.items():
        tensor = state['model'].get(name)
        if tensor is None or list(tensor.shape) != specification['shape']:
            raise ValueError(f'Recovery tensor missing or wrong shape: {name}')
        if specification['trainable'] and not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite recovery tensor: {name}')
    optimizer = state['optimizer']
    if not optimizer['state'] or any(int(s['step']) != 587 for s in optimizer['state'].values()):
        raise ValueError('Recovery optimizer progress is not e1/587 updates')
    for group in optimizer['param_groups']:
        if group['initial_lr'] not in (1e-4, 3e-4) or not math.isclose(
                group['lr'], group['initial_lr'], rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError('Recovery learning rate is not the end-of-warmup rate')
    return dict(time=stamp(), source_suite=str(source), checkpoint=str(checkpoint),
                checkpoint_sha256=recovery['checkpoint_sha256'], epoch=1, global_step=587,
                training_images=9392, optimizer_and_rank_rng_saved=True,
                allocator_config=recovery['allocator_config'])


def verify_snapshot(suite, dependency, data):
    code, previous = suite / 'code', dependency / 'code'
    manifest, earlier = read_json(suite / 'launch_manifest.json'), read_json(dependency / 'launch_manifest.json')
    entries = (suite / 'source.sha256').read_text().splitlines()
    if not entries:
        raise ValueError('Empty frozen source manifest')
    for line in entries:
        expected, name = line.split('  ', 1)
        source = (code / name).resolve()
        if not source.is_relative_to(code.resolve()) or sha256_file(source) != expected:
            raise ValueError(f'Frozen source checksum mismatch: {name}')
    if not manifest.get('data_sha256') or manifest['data_sha256'] != earlier['data_sha256']:
        raise ValueError('Data manifests differ from completed BP7')
    for name, expected in manifest['data_sha256'].items():
        source = (data / name).resolve()
        if not source.is_relative_to(data.resolve()) or sha256_file(source) != expected:
            raise ValueError(f'Data checksum mismatch: {name}')
    shared = {'train.py', 'predict.py', 'pp_eval.py', 'pp_study.py', 'make_submission.py', 'score_submission.py'}
    shared |= {str(p.relative_to(previous)) for p in (previous / 'sarbench').rglob('*.py')}
    for name in sorted(shared):
        if sha256_file(code / name) != sha256_file(previous / name):
            raise ValueError(f'Training/model implementation changed: {name}')
    if (manifest['expected_parameters'], manifest['expected_trainable_parameters']) != EXPECTED_BUDGET[:2]:
        raise ValueError('Unexpected BP7 architecture budget')
    if (manifest['epochs'], manifest['schedule_epochs'], manifest['stop_after_epoch']) != (200, 200, 160):
        raise ValueError('Unexpected training duration or cosine schedule')
    if manifest['gpu_memory_fraction'] != MEMORY_FRACTION:
        raise ValueError('Unexpected concurrent GPU memory budget')
    outputs = manifest['submission_paths']
    if set(outputs) != {'130', '160'} or len({str(Path(p).resolve()) for p in outputs.values()}) != 2:
        raise ValueError('Need two distinct e130/e160 submission destinations')
    return dict(source_files=len(entries), matched_implementation_files=len(shared),
                data_manifests=len(manifest['data_sha256']))


def verify_stopped_run(run, command, epoch, images, *, expected_budget=EXPECTED_BUDGET):
    """Verify genuine 200-cosine progress, weights, optimizer and two-rank RNG."""
    import torch
    import train
    from predict import saved_args

    expected = vars(train.parse_args(command[command.index('train.py') + 1:]))
    expected = json.loads(json.dumps(expected, default=str))
    if expected['epochs'] != 200 or expected['schedule_epochs'] != 200:
        raise ValueError('Expected the unchanged 200-epoch cosine')
    if saved_args(run) != expected:
        raise ValueError('Saved arguments differ from the declared stage command')
    log_text = (run / 'log.txt').read_text()
    if f'Stopped cleanly after epoch {epoch};' not in log_text:
        raise ValueError('Training did not stop cleanly at the requested epoch')
    state = torch.load(run / 'training_state.pt', map_location='cpu', weights_only=False)
    signature, steps = state['signature'], math.ceil(images / 16)
    if (state.get('format_version') != 1 or state['epoch'] != epoch
            or state['global_step'] != epoch * steps or state['scheduler']['last_epoch'] != epoch * steps):
        raise ValueError('Checkpoint epoch, updates and scheduler progress disagree')
    if (signature['world_size'] != 2 or signature['dataset_size'] != images
            or signature['steps_per_epoch'] != steps or len(state['rank_rng']) != 2
            or state['ema'] is not None or state['history']):
        raise ValueError('Unexpected data coverage, ranks, EMA or evaluation')
    for key, value in signature['args'].items():
        if key not in expected or expected[key] != value:
            raise ValueError(f'Checkpoint configuration mismatch: {key}')
    for name, digest in signature['manifests'].items():
        if sha256_file(Path(expected['data']) / name) != digest:
            raise ValueError(f'Training manifest changed: {name}')
    parameters = signature['parameters']
    total = sum(math.prod(p['shape']) for p in parameters.values())
    trainable = sum(math.prod(p['shape']) for p in parameters.values() if p['trainable'])
    persistent = sum(t.numel() for k, t in state['model'].items()
                     if k not in parameters and t.is_floating_point())
    budget = re.search(r'parameters: ([\d,]+) total, ([\d,]+) trainable '
                       r'\(\+ ([\d,]+) floating-point buffer values\)', log_text)
    if budget is None:
        raise ValueError('Missing runtime parameter budget')
    reported = tuple(int(v.replace(',', '')) for v in budget.groups())
    if (total, trainable, reported[2]) != tuple(expected_budget) or reported[:2] != (total, trainable):
        raise ValueError('Declared parameter budget mismatch')
    if total + reported[2] > 130000000 or trainable > 40000000:
        raise ValueError('Model exceeds parameter budget')
    weights = run / f'ckpt_e{epoch}.pt'
    inference = torch.load(weights, map_location='cpu', weights_only=True)
    if not {n for n, p in parameters.items() if p['trainable']}.issubset(inference):
        raise ValueError('Inference checkpoint lacks trained parameters')
    for name, tensor in inference.items():
        if name not in state['model'] or not torch.equal(tensor, state['model'][name]):
            raise ValueError(f'Inference and resumable weights differ: {name}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite trained tensor: {name}')
    optimizer = state['optimizer']
    if not optimizer['state'] or any(int(s['step']) != epoch * steps for s in optimizer['state'].values()):
        raise ValueError('Optimizer progress differs from checkpoint epoch')
    factor = train.lr_factor(epoch * steps, round(expected['warmup_epochs'] * steps), 200 * steps)
    for group in optimizer['param_groups']:
        if group['initial_lr'] not in (1e-4, 3e-4) or not math.isclose(
                group['lr'], group['initial_lr'] * factor, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError('Learning rate does not follow the 200-epoch cosine')
    return dict(time=stamp(), epoch=epoch, schedule_epochs=200, training_images=images,
                global_step=state['global_step'], steps_per_epoch=steps, parameters=total,
                trainable_parameters=trainable, buffer_values=reported[2], persistent_buffer_values=persistent,
                lr_factor=factor, train_hours=state['train_hours'], training_gpu_hours=2 * state['train_hours'],
                checkpoint=str(weights), checkpoint_sha256=sha256_file(weights),
                training_state_sha256=sha256_file(run / 'training_state.pt'),
                trained_tensors_match=True, optimizer_and_rank_rng_saved=True)


def verify_data(data, official):
    parts = [read_json(data / 'detection' / f'instances_{split}.json')['images'] for split in ('train', 'val', 'test')]
    names = [Path(im['file_name']).name for part in parts for im in part]
    original = read_json(official / 'train/instances.json')['images']
    val_images = read_json(official / 'val/images.json')
    if (len(names) != 9392 or len(set(names)) != 9392
            or set(names) != {Path(im['file_name']).name for im in original}
            or len(val_images['images']) != 1426):
        raise ValueError('Training union or official prediction coverage mismatch')
    original_by_id = {im['id']: im for im in original}
    if {im['id'] for part in parts for im in part} != set(original_by_id):
        raise ValueError('Converted image IDs do not match official training IDs')
    for part in parts:
        for im in part:
            source = official / 'train' / original_by_id[im['id']]['file_name']
            if not source.is_file() or (data / im['file_name']).resolve() != source.resolve():
                raise ValueError('Converted image path is not an official training image')
    return val_images, dict(count=9392, split_counts=list(map(len, parts)), disjoint=True,
                           matches_official_training=True, resolved_training_paths_verified=True,
                           official_val_used_for_training=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    if Path(__file__).resolve().parents[1] != suite / 'code':
        raise RuntimeError('Run only from the frozen source snapshot')
    lock = (suite / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite / 'status.json').exists():
        raise RuntimeError('Refusing to overwrite an existing suite')
    manifest = read_json(suite / 'launch_manifest.json')
    data, official = Path(manifest['data']), Path(manifest['official_data'])
    dependency = Path(manifest['dependency_suite'])
    recovery = manifest.get('recovery')
    allocator_config = recovery['allocator_config'] if recovery else None
    configure_resources(allocator_config)
    state = dict(pid=os.getpid(), started_at=stamp(), suite=str(suite), gpus=[0, 1],
                 cpu_affinity=sorted(os.sched_getaffinity(0)), global_batch=16, schedule_epochs=200,
                 stop_after_epoch=160, training_images=9392, maximum_new_training_runs=1,
                 official_submissions=0, gpu_memory_fraction=MEMORY_FRACTION,
                 allocator_config=allocator_config, submissions={})
    commands = {}

    def update(stage=None, **extra):
        if stage:
            state['stage'] = stage
        state.update(updated_at=stamp(), **extra)
        write_json(suite / 'status.json', state)
        if stage:
            print(stamp(), stage, extra, flush=True)

    def execute(name, command, log, single_gpu=False):
        commands[name] = command
        write_json(suite / 'commands.json', commands)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1') if single_gpu else None
        with log.open('a') as stream:
            with subprocess.Popen(command, cwd=suite / 'code', env=env,
                                  stdout=stream, stderr=subprocess.STDOUT) as child:
                update(child_pid=child.pid)
                while True:
                    try:
                        rc = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        update()
        state.pop('child_pid', None)
        update()
        if rc:
            raise subprocess.CalledProcessError(rc, command)

    def predict_command(run, epoch, output, images):
        return [sys.executable, 'tools/train_gpu_limited.py', '--memory-fraction', str(MEMORY_FRACTION),
                'predict.py', '--run', str(run), '--weights', f'ckpt_e{epoch}.pt',
                '--images', str(official / 'val'), '--images-json', str(images),
                '--batch-size', '2', '--workers', '0', '--shift', '-0.2', '--out', str(output)]

    try:
        update('verifying_inputs')
        write_json(suite / 'input_verification.json', verify_snapshot(suite, dependency, data))
        val_images, data_check = verify_data(data, official)
        write_json(suite / 'full_data_check.json', data_check)
        outputs = {int(epoch): Path(path) for epoch, path in manifest['submission_paths'].items()}
        if any(path.exists() for path in outputs.values()):
            raise RuntimeError('Refusing to overwrite an existing submission')
        validation = suite / 'validation'
        validation.mkdir()
        if recovery:
            update('verifying_recovery')
            recovered = verify_recovery(manifest, data)
            write_json(suite / 'recovery_input_validation.json', recovered)
            smoke = validation / 'resume_probe'
            smoke_epoch = 2
            smoke_command = recovery_probe_command(sys.executable, data, validation,
                                                    Path(recovery['checkpoint']))
            update('recovery_validation', active_trial='resume_probe', target_epoch=2)
            execute('resume_probe', smoke_command, validation / 'resume_probe_train.log')
            smoke_result = verify_stopped_run(smoke, smoke_command, 2, 9392)
            smoke_result['recovery'] = recovered
            smoke_result['allocator_config'] = allocator_config
            previous_state = smoke / 'training_state.pt'
        else:
            smoke = validation / 'smoke'
            smoke_epoch = 1
            smoke_command = training_command(sys.executable, data, validation, 'smoke', stop_epoch=1)
            for flag, value in {'--save-every': '1', '--workers': '0'}.items():
                smoke_command[smoke_command.index(flag) + 1] = value
            smoke_command += ['--limit', '48']
            update('smoke_training')
            execute('smoke', smoke_command, validation / 'smoke_train.log')
            smoke_result = verify_stopped_run(smoke, smoke_command, 1, 48)
            previous_state = None
        image_list = validation / 'smoke_images.json'
        write_json(image_list, dict(images=val_images['images'][:8]))
        smoke_zip = validation / 'smoke_submission.zip'
        update('smoke_prediction')
        execute('smoke_prediction', predict_command(smoke, smoke_epoch, smoke_zip, image_list),
                validation / 'smoke_predict.log', single_gpu=True)
        smoke_result['submission'] = validate_zip(smoke_zip, image_list)
        write_json(validation / 'smoke_passed.json', smoke_result)
        if recovery:
            write_json(validation / 'recovery_passed.json', smoke_result)
        for epoch, name in [(130, 'full_train'), (160, 'full_train_e160')]:
            verify_snapshot(suite, dependency, data)
            run = suite / name
            run.mkdir()
            command = training_command(sys.executable, data, suite, name,
                                       stop_epoch=epoch, resume=previous_state)
            update('training', active_trial=name, target_epoch=epoch)
            execute(name, command, run / 'stdout.log')
            update(f'verifying_epoch{epoch}')
            result = verify_stopped_run(run, command, epoch, 9392)
            write_json(suite / f'checkpoint_e{epoch}_validation.json', result)
            update(f'generating_submission_e{epoch}')
            output = outputs[epoch]
            if output.exists():
                raise RuntimeError('Refusing to overwrite an existing submission')
            execute(f'submission_e{epoch}', predict_command(run, epoch, output, official / 'val/images.json'),
                    run / 'submission.log', single_gpu=True)
            validated = validate_zip(output, official / 'val/images.json')
            validated.update(time=stamp(), checkpoint_sha256=result['checkpoint_sha256'],
                             epoch=epoch, schedule_epochs=200, training_images=9392)
            write_json(suite / f'submission_e{epoch}_validation.json', validated)
            output.with_suffix('.sha256').write_text(validated['sha256'] + '  ' + output.name + '\n')
            state['submissions'][str(epoch)] = str(output)
            previous_state = run / 'training_state.pt'
            # The continuation lives in a separate directory; the complete e130
            # optimizer/scheduler/RNG state is never replaced by later epochs.
            update(f'completed_epoch{epoch}', active_trial=None, train_hours=result['train_hours'])
        update('complete', active_trial=None)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
