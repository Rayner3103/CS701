"""Queue one LingBot-Vision-B BP7 trial after the current BP7 finishes and exports."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from run_bp7_schedule_search import configure_resources, read_json, sha256_file, stamp, write_json
from run_bp7_lr125_trial import process_identity
from run_bp7_full_stop130 import (
    training_command as bp7_training_command, resume_command, validate_zip,
    verify_stopped_run as verify_bp7_stopped_run,
)


def training_command(python, data, suite, name='full_train'):
    command = bp7_training_command(python, data, suite, name)
    command[command.index('--backbone') + 1] = 'lingbot_b'
    # CUDA compiled blocks failed non-reentrant checkpoint recomputation checks.
    # Keep eager blocks and checkpointing; retain all gradient safety checks.
    command.remove('--compile')
    return command


def verify_stopped_run(run, command, epoch, images):
    suite = run.parent.parent if run.parent.name == 'validation' else run.parent
    manifest = read_json(suite / 'launch_manifest.json')
    budget = (manifest['expected_parameters'], manifest['expected_trainable_parameters'],
              manifest['expected_floating_buffers'])
    return verify_bp7_stopped_run(run, command, epoch, images, expected_budget=budget)


def verify_snapshot(suite, dependency, data):
    code, previous = suite / 'code', dependency / 'code'
    manifest, earlier = read_json(suite / 'launch_manifest.json'), read_json(dependency / 'launch_manifest.json')
    entries = (suite / 'source.sha256').read_text().splitlines()
    if not entries:
        raise ValueError('Empty source manifest')
    for line in entries:
        expected, name = line.split('  ', 1)
        source = (code / name).resolve()
        if not source.is_relative_to(code.resolve()) or sha256_file(source) != expected:
            raise ValueError(f'Frozen source checksum mismatch: {name}')
    if manifest['data_sha256'] != earlier['data_sha256']:
        raise ValueError('The LingBot and BP7 data manifests differ')
    for name, expected in manifest['data_sha256'].items():
        source = (data / name).resolve()
        if not source.is_relative_to(data.resolve()) or sha256_file(source) != expected:
            raise ValueError(f'Data checksum mismatch: {name}')
    shared = {'train.py', 'predict.py', 'pp_eval.py', 'pp_study.py', 'make_submission.py', 'score_submission.py'}
    shared |= {str(p.relative_to(previous)) for p in (previous / 'sarbench').rglob('*.py')}
    allowed_changes = {'sarbench/backbones.py', 'sarbench/adapters.py'}
    for name in sorted(shared - allowed_changes):
        if sha256_file(code / name) != sha256_file(previous / name):
            raise ValueError(f'Unplanned training/head change: {name}')
    weights = manifest['pretrained']
    if Path(weights['path']).stat().st_size != weights['bytes'] or sha256_file(weights['path']) != weights['sha256']:
        raise ValueError('Generic pretrained weight checksum mismatch')
    if manifest['expected_parameters'] + manifest['expected_floating_buffers'] > 130000000 or manifest['expected_trainable_parameters'] > 40000000:
        raise ValueError('Model exceeds the competition parameter budget')
    if manifest['epochs'] != 160 or manifest['schedule_epochs'] != 160 or manifest['stop_after_epoch'] != 130:
        raise ValueError('Unexpected training duration/schedule')
    return dict(source_files=len(entries), matched_implementation_files=len(shared - allowed_changes),
                allowed_backbone_integration_files=sorted(allowed_changes), pretrained_sha256=weights['sha256'],
                data_manifests=len(manifest['data_sha256']))


def wait_for_previous(previous, update):
    """Run no GPU work until the predecessor has exported and released its lock."""
    deadline = time.monotonic() + 48 * 3600
    with (previous / 'driver.lock').open('r') as lock:
        while True:
            state = read_json(previous / 'status.json')
            if state['stage'] in ('failed', 'stopped_by_user', 'superseded'):
                raise RuntimeError(f'Predecessor did not finish normally: {state["stage"]}')
            identity = process_identity(state['pid'])
            alive = identity is not None and str(previous / 'code/tools/run_bp7_full_stop130.py') in identity['command']
            if state['stage'] == 'complete':
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    checkpoint = read_json(previous / 'checkpoint_validation.json')
                    exported = read_json(previous / 'submission_validation.json')
                    if (checkpoint['epoch'] != 130 or checkpoint['global_step'] != 76310
                            or checkpoint['training_images'] != 9392 or exported['epoch'] != 130
                            or exported['prediction_images'] != 1426 or not exported['format_passed']
                            or exported['checkpoint_sha256'] != checkpoint['checkpoint_sha256']
                            or sha256_file(exported['path']) != exported['sha256']):
                        raise ValueError('Predecessor artifacts do not match its declared e130 export')
                    return dict(time=stamp(), previous_suite=str(previous),
                                previous_submission=exported['path'], previous_submission_sha256=exported['sha256'])
            elif not alive:
                raise RuntimeError('Predecessor died before completion')
            if time.monotonic() > deadline:
                raise RuntimeError('Wait for predecessor exceeded 48 hours')
            update(dependency_stage=state['stage'])
            time.sleep(30)


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
    output = Path(manifest['submission_path'])
    configure_resources()
    state = dict(pid=os.getpid(), started_at=stamp(), suite=str(suite), gpus=[0, 1],
                 cpu_affinity=sorted(os.sched_getaffinity(0)), global_batch=16,
                 schedule_epochs=160, stop_after_epoch=130, training_images=9392,
                 maximum_new_training_runs=1, official_submissions=0, backbone='lingbot_b',
                 wait_for_suite=manifest['wait_for_suite'])
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
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0') if single_gpu else None
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

    def predict_command(run, epoch, destination, image_list):
        return [sys.executable, 'predict.py', '--run', str(run), '--weights', f'ckpt_e{epoch}.pt',
                '--images', str(official / 'val'), '--images-json', str(image_list),
                '--batch-size', '2', '--workers', '0', '--shift', '-0.2', '--out', str(destination)]

    try:
        update('verifying_inputs')
        checks = verify_snapshot(suite, Path(manifest['dependency_suite']), data)
        write_json(suite / 'input_verification.json', checks)
        parts = [read_json(data / 'detection' / f'instances_{split}.json')['images']
                 for split in ('train', 'val', 'test')]
        names = [Path(im['file_name']).name for part in parts for im in part]
        original = read_json(official / 'train/instances.json')['images']
        val_images = read_json(official / 'val/images.json')
        if (len(names) != 9392 or len(set(names)) != 9392
                or set(names) != {Path(im['file_name']).name for im in original}
                or len(val_images['images']) != 1426):
            raise ValueError('Training union or official prediction coverage mismatch')
        # Official train and val intentionally reuse basenames in separate folders.
        # Cross-split basename overlap is not evidence of shared image content.
        original_by_id = {im['id']: im for im in original}
        if {im['id'] for part in parts for im in part} != set(original_by_id):
            raise ValueError('Converted image IDs do not match official training IDs')
        for part in parts:
            for im in part:
                source = official / 'train' / original_by_id[im['id']]['file_name']
                if not source.is_file() or (data / im['file_name']).resolve() != source.resolve():
                    raise ValueError('Converted image path is not the corresponding official training image')
        write_json(suite / 'full_data_check.json', dict(count=9392, split_counts=list(map(len, parts)),
                   disjoint=True, matches_official_training=True, resolved_training_paths_verified=True,
                   official_val_used_for_training=False))
        if output.exists():
            raise RuntimeError('Refusing to overwrite an existing submission')
        update('waiting_for_bp7')
        handoff = wait_for_previous(Path(manifest['wait_for_suite']), update)
        write_json(suite / 'dependency_completion.json', handoff)
        verify_snapshot(suite, Path(manifest['dependency_suite']), data)
        validation = suite / 'validation'
        validation.mkdir()
        smoke = validation / 'smoke'
        smoke_command = training_command(sys.executable, data, validation, 'smoke')
        for flag, value in {'--stop-after-epoch': '1', '--save-every': '1', '--workers': '0'}.items():
            smoke_command[smoke_command.index(flag) + 1] = value
        smoke_command += ['--limit', '48']
        update('smoke_training')
        execute('smoke', smoke_command, validation / 'smoke_train.log')
        smoke_result = verify_stopped_run(smoke, smoke_command, 1, 48)
        image_list = validation / 'smoke_images.json'
        write_json(image_list, dict(images=val_images['images'][:8]))
        smoke_zip = validation / 'smoke_submission.zip'
        update('smoke_prediction')
        execute('smoke_prediction', predict_command(smoke, 1, smoke_zip, image_list),
                validation / 'smoke_predict.log', single_gpu=True)
        smoke_result['submission'] = validate_zip(smoke_zip, image_list)
        write_json(validation / 'smoke_passed.json', smoke_result)
        verify_snapshot(suite, Path(manifest['dependency_suite']), data)
        run = suite / 'full_train'
        run.mkdir()
        command = training_command(sys.executable, data, suite)
        write_json(suite / 'resume_command.json', dict(cwd=str(suite / 'code'),
                   command=resume_command(command, run / 'training_state.pt'),
                   note='Manual continuation after e130, not automatically executed; keep the original environment.'))
        update('training', active_trial='full_train')
        execute('full_train', command, run / 'stdout.log')
        update('verifying_epoch130')
        result = verify_stopped_run(run, command, 130, 9392)
        write_json(suite / 'checkpoint_validation.json', result)
        update('generating_submission')
        execute('submission', predict_command(run, 130, output, official / 'val/images.json'),
                run / 'submission.log', single_gpu=True)
        update('checking_submission')
        validated = validate_zip(output, official / 'val/images.json')
        validated.update(time=stamp(), checkpoint_sha256=result['checkpoint_sha256'],
                         epoch=130, schedule_epochs=160, training_images=9392)
        write_json(suite / 'submission_validation.json', validated)
        output.with_suffix('.sha256').write_text(validated['sha256'] + '  ' + output.name + '\n')
        update('complete', active_trial=None, submission=str(output),
               train_hours=result['train_hours'], checkpoint_sha256=result['checkpoint_sha256'])
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
