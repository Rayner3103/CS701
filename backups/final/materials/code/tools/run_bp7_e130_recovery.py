"""Resume the verified BP7 cosine-200 e130 state through an e131 probe to e160."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from run_bp7_cos200_final import configure_resources, validate_zip, verify_data, verify_stopped_run
from run_bp7_schedule_search import read_json, sha256_file, stamp, write_json

MEMORY_FRACTION = 0.35
ALLOCATOR_CONFIG = 'expandable_segments:True'
MUTABLE_FLAGS = ('--memory-fraction', '--out', '--name', '--stop-after-epoch', '--save-every', '--resume')


def value(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f'Expected exactly one {flag}')
    return command[command.index(flag) + 1]


def continuation_command(template, out, name, epoch, checkpoint, save_every):
    if epoch not in (131, 160):
        raise ValueError('Only e131 probe and e160 continuation are authorized')
    command = list(template)
    for flag, replacement in zip(MUTABLE_FLAGS, (
            MEMORY_FRACTION, out, name, epoch, save_every, checkpoint)):
        value(command, flag)
        command[command.index(flag) + 1] = str(replacement)
    return command


def verify_frozen(suite):
    code = (suite / 'code').resolve()
    entries = {}
    for line in (suite / 'source.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (code / name).resolve()
        if name in entries or not path.is_relative_to(code) or sha256_file(path) != digest:
            raise ValueError(f'Frozen source checksum mismatch: {name}')
        entries[name] = digest
    if not entries:
        raise ValueError('Empty frozen source manifest')
    return entries


def verify_inputs(suite, manifest):
    source = Path(manifest['source_suite']).resolve()
    if source == suite or suite.is_relative_to(source):
        raise ValueError('Recovery must use a separate suite')
    if manifest['gpu_memory_fraction'] != MEMORY_FRACTION or manifest['allocator_config'] != ALLOCATOR_CONFIG:
        raise ValueError('Unexpected recovery resource settings')
    current, previous = verify_frozen(suite), verify_frozen(source)
    required = {'train.py', 'predict.py', 'score_submission.py', 'tools/train_gpu_limited.py',
                'tools/run_bp7_cos200_final.py', 'tools/run_bp7_full_stop130.py',
                'tools/run_bp7_schedule_search.py', 'tools/run_bp7_lr_trials.py'}
    if not required.issubset(previous) or not required.union({'tools/run_bp7_e130_recovery.py'}).issubset(current):
        raise ValueError('Frozen manifest lacks a runtime dependency')
    for name, digest in previous.items():
        if name == 'tools/train_gpu_limited.py':
            expected = (source / 'code' / name).read_text().replace('0.20', '0.35')
            if (suite / 'code' / name).read_text() != expected:
                raise ValueError('GPU wrapper changed beyond the approved cap')
        elif current.get(name) != digest:
            raise ValueError(f'Frozen training implementation changed: {name}')
    if read_json(source / 'status.json')['stage'] != 'failed':
        raise ValueError('Source controller must be failed and stopped')
    original = read_json(source / 'launch_manifest.json')
    for key in ('data', 'official_data'):
        if not Path(manifest[key]).is_absolute() or Path(manifest[key]).resolve() != Path(original[key]).resolve():
            raise ValueError(f'Recovery {key} differs from the original run')
    data = Path(manifest['data'])
    for name, digest in original['data_sha256'].items():
        path = (data / name).resolve()
        if not path.is_relative_to(data.resolve()) or sha256_file(path) != digest:
            raise ValueError(f'Training data checksum mismatch: {name}')
    commands = read_json(source / 'commands.json')
    template, original130 = commands['full_train_e160'], commands['full_train']
    prefix = [template[0], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
              'tools/train_gpu_limited.py', '--memory-fraction', '0.2', 'train.py']
    if template[:10] != prefix or original130[:10] != prefix or not Path(template[0]).is_absolute():
        raise ValueError('Unexpected original distributed command')
    expected = {'--epochs': '200', '--schedule-epochs': '200', '--batch-size': '16',
                '--workers': '1', '--lr': '0.0001', '--backbone-lr': '0.0003',
                '--train-split': 'train+val+test', '--eval-splits': '', '--seed': '0',
                '--data': str(data), '--out': str(source), '--name': 'full_train_e160',
                '--stop-after-epoch': '160', '--save-every': '10',
                '--resume': str(source / 'full_train/training_state.pt')}
    for flag, expected_value in expected.items():
        if value(template, flag) != expected_value:
            raise ValueError(f'Unexpected original training argument: {flag}')
    if any(flag in template for flag in ('--compile', '--limit')):
        raise ValueError('Unexpected compilation or data limit')
    if not all(flag in template for flag in ('--grad-ckpt', '--save-training-state', '--reproducible-loader')):
        raise ValueError('Missing required full-state training flags')
    def immutable(command):
        result = list(command)
        for flag in MUTABLE_FLAGS:
            value(result, flag)
            position = result.index(flag)
            del result[position:position + 2]
        return result
    if immutable(template) != immutable(original130):
        raise ValueError('Original e130 and e160 recipes differ')
    if (value(original130, '--out') != str(source) or value(original130, '--name') != 'full_train'
            or value(original130, '--stop-after-epoch') != '130'):
        raise ValueError('Original e130 command does not describe the saved run')
    checkpoint = source / 'full_train/training_state.pt'
    if sha256_file(checkpoint) != manifest['source_checkpoint_sha256']:
        raise ValueError('Source e130 checkpoint checksum mismatch')
    existing = Path(manifest['existing_e130_submission'])
    output = Path(manifest['submission_path'])
    if not existing.is_absolute() or not output.is_absolute() or existing.resolve() == output.resolve():
        raise ValueError('Need distinct absolute e130 and e160 submission paths')
    if sha256_file(existing) != manifest['existing_e130_submission_sha256']:
        raise ValueError('Existing e130 submission checksum mismatch')
    if output.exists() or output.with_suffix('.sha256').exists():
        raise FileExistsError('Refusing to overwrite an existing e160 submission')
    checkpoint_check = verify_stopped_run(source / 'full_train', original130, 130, 9392)
    submission_check = validate_zip(existing, Path(manifest['official_data']) / 'val/images.json')
    return template, dict(time=stamp(), source_suite=str(source), source_files=len(previous),
                         frozen_files=len(current), allocator_config=ALLOCATOR_CONFIG,
                         gpu_memory_fraction=MEMORY_FRACTION, checkpoint=checkpoint_check,
                         existing_e130_submission=submission_check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    suite = parser.parse_args().suite.resolve()
    if Path(__file__).resolve().parents[1] != suite / 'code':
        raise RuntimeError('Run only from the frozen source snapshot')
    with (suite / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (suite / 'status.json').exists():
            raise FileExistsError('Refusing to overwrite an existing recovery suite')
        manifest = read_json(suite / 'launch_manifest.json')
        if manifest['allocator_config'] != ALLOCATOR_CONFIG:
            raise ValueError('Unexpected recovery allocator configuration')
        configure_resources(ALLOCATOR_CONFIG)
        state = dict(pid=os.getpid(), started_at=stamp(), suite=str(suite), source_suite=manifest['source_suite'],
                     gpus=[0, 1], cpu_affinity=sorted(os.sched_getaffinity(0)), global_batch=16,
                     training_images=9392, schedule_epochs=200, stop_after_epoch=160,
                     gpu_memory_fraction=MEMORY_FRACTION, allocator_config=ALLOCATOR_CONFIG,
                     official_submissions=0, submissions={'130': manifest['existing_e130_submission']})
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
            with log.open('x') as stream:
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

        def prediction(run, epoch, images, output):
            if output.exists():
                raise FileExistsError(f'Refusing to overwrite {output}')
            return [sys.executable, 'tools/train_gpu_limited.py', '--memory-fraction', str(MEMORY_FRACTION),
                    'predict.py', '--run', str(run), '--weights', f'ckpt_e{epoch}.pt',
                    '--images', str(Path(manifest['official_data']) / 'val'), '--images-json', str(images),
                    '--batch-size', '2', '--workers', '0', '--shift', '-0.2', '--out', str(output)]

        try:
            update('verifying_inputs')
            source = Path(manifest['source_suite']).resolve()
            # Hold the original controller lock until recovery ends, preventing
            # an old driver from modifying the verified source during this run.
            with (source / 'driver.lock').open('r') as source_lock:
                fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                template, inputs = verify_inputs(suite, manifest)
                write_json(suite / 'input_verification.json', inputs)
                val_images, data_check = verify_data(Path(manifest['data']), Path(manifest['official_data']))
                write_json(suite / 'full_data_check.json', data_check)
                validation = suite / 'validation'
                validation.mkdir()
                probe = validation / 'resume_e131'
                probe.mkdir()
                command = continuation_command(template, validation, 'resume_e131', 131,
                                               source / 'full_train/training_state.pt', 1)
                update('probe_training', active_trial='resume_e131', target_epoch=131)
                execute('resume_e131', command, probe / 'stdout.log')
                update('verifying_probe')
                probe_result = verify_stopped_run(probe, command, 131, 9392)
                write_json(suite / 'checkpoint_e131_validation.json', probe_result)
                image_list = validation / 'probe_images.json'
                write_json(image_list, dict(images=val_images['images'][:8]))
                probe_zip = validation / 'probe_submission.zip'
                update('probe_prediction')
                execute('probe_prediction', prediction(probe, 131, image_list, probe_zip),
                        validation / 'probe_prediction.log', single_gpu=True)
                probe_result.update(submission=validate_zip(probe_zip, image_list),
                                    allocator_config=ALLOCATOR_CONFIG, gpu_memory_fraction=MEMORY_FRACTION)
                write_json(validation / 'probe_passed.json', probe_result)
                # Recheck both code snapshots after the GPU probe before the long continuation.
                verify_frozen(suite)
                verify_frozen(source)
                run = suite / 'full_train_e160'
                run.mkdir()
                command = continuation_command(template, suite, 'full_train_e160', 160,
                                               probe / 'training_state.pt', 10)
                update('training', active_trial='full_train_e160', target_epoch=160)
                execute('full_train_e160', command, run / 'stdout.log')
                update('verifying_epoch160')
                result = verify_stopped_run(run, command, 160, 9392)
                write_json(suite / 'checkpoint_e160_validation.json', result)
                output = Path(manifest['submission_path'])
                if output.with_suffix('.sha256').exists():
                    raise FileExistsError('Refusing to overwrite a submission checksum')
                images = Path(manifest['official_data']) / 'val/images.json'
                update('generating_submission_e160')
                execute('submission_e160', prediction(run, 160, images, output),
                        run / 'submission.log', single_gpu=True)
                checked = validate_zip(output, images)
                checked.update(time=stamp(), epoch=160, schedule_epochs=200, training_images=9392,
                               checkpoint_sha256=result['checkpoint_sha256'])
                write_json(suite / 'submission_e160_validation.json', checked)
                with output.with_suffix('.sha256').open('x') as stream:
                    stream.write(checked['sha256'] + '  ' + output.name + '\n')
                state['submissions']['160'] = str(output)
                update('complete', active_trial=None, train_hours=result['train_hours'])
        except Exception as error:
            update('failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
