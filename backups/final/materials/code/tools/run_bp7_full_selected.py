"""Finish local selection, train one full-data BP7 model, and export a validated submission."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from run_bp7_lr125_trial import process_identity
from run_bp7_schedule_search import (
    build_training_command, configure_resources, dependency_ready, evaluation_command,
    read_json, stamp, validate_result, verify_snapshot, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    manifest = read_json(suite / 'launch_manifest.json')
    baseline_suite = Path(manifest['dependency_suite'])
    previous = Path(manifest['wait_for_suite'])
    data, reference = Path(manifest['data']), Path(manifest['reference'])
    if Path(__file__).resolve().parents[1] != suite / 'code':
        raise RuntimeError('Use the frozen snapshot')
    lock = (suite / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite / 'status.json').exists():
        raise RuntimeError('Refusing to overwrite an existing suite')
    configure_resources()
    state = dict(pid=os.getpid(), started_at=stamp(), suite=str(suite),
                 wait_for_suite=str(previous), maximum_new_training_runs=1,
                 gpus=[0, 1], cpu_affinity=sorted(os.sched_getaffinity(0)), global_batch=16,
                 official_submissions=0, automatic_full_data_retrain=True)
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
        with log.open('a') as output:
            with subprocess.Popen(command, cwd=suite / 'code', env=env,
                                  stdout=output, stderr=subprocess.STDOUT) as child:
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

    dependency_lock = None
    try:
        update('verifying_inputs')
        checks = verify_snapshot(suite, baseline_suite, data, reference)
        write_json(suite / 'input_verification.json', checks)
        baseline_run = baseline_suite / 'lr1x'
        baseline = validate_result(baseline_run, baseline_run / 'metrics_submission_post_val.json',
                                   data=data, reference=reference, epochs=130, head_lr=1e-4, backbone_lr=3e-4)
        update('waiting_for_dependency')
        dependency_lock = (previous / 'driver.lock').open('r')
        deadline = time.monotonic() + 48 * 3600
        while True:
            old = read_json(previous / 'status.json')
            acquired = False
            cancelled = (old['stage'] == 'stopped_by_user'
                         and manifest.get('allow_user_stopped_dependency') is True)
            if old['stage'] == 'complete' or cancelled:
                try:
                    fcntl.flock(dependency_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    pass
            identity = process_identity(old['pid'])
            expected_driver = str(previous / 'code/tools/run_bp7_fixed160.py')
            alive = identity is not None and expected_driver in identity['command']
            if cancelled:
                if alive:
                    raise RuntimeError('Cancelled dependency controller still running')
                for proc in Path('/proc').iterdir():
                    if proc.name.isdigit():
                        item = process_identity(int(proc.name))
                        if item and 'train.py' in item['command'] and str(previous) in item['command']:
                            raise RuntimeError('Cancelled dependency still has training processes')
                if acquired:
                    break
                raise RuntimeError('Cancelled dependency still holds its lock')
            if dependency_ready(old, locked=not acquired, alive=alive):
                break
            if old['stage'] == 'superseded' or time.monotonic() > deadline:
                raise RuntimeError('Dependency superseded or wait exceeded 48 hours')
            update(dependency_stage=old['stage'])
            time.sleep(30)
        verify_snapshot(suite, baseline_suite, data, reference)
        local_run = previous / 'schedule160'
        local = validate_result(local_run, local_run / 'metrics_submission_post_val.json',
                                data=data, reference=reference, epochs=160, head_lr=1e-4, backbone_lr=3e-4)
        gain = local['local_delta_m'] - baseline['local_delta_m']
        epochs = 160 if gain > 1.0 else 130
        write_json(suite / 'selection.json', dict(time=stamp(), epochs=epochs,
                   baseline_local_delta_m=baseline['local_delta_m'],
                   schedule160_local_delta_m=local['local_delta_m'], gain=gain,
                   policy='Use160 only if gain>1.0 local delta_m; otherwise130. Operational, not statistical.',
                   source='Final local validation, identical submission postprocessing',
                   seed_confirmation_skipped='User authorized proceeding directly to full train overnight'))
        official = Path(manifest['official_data'])
        splits = [read_json(data / 'detection' / f'instances_{split}.json')['images']
                  for split in ('train', 'val', 'test')]
        names = [im['file_name'] for images in splits for im in images]
        original = read_json(official / 'train/instances.json')['images']
        assert len(names) == len(set(names)) == 9392
        assert {Path(n).name for n in names} == {Path(im['file_name']).name for im in original}
        write_json(suite / 'full_data_check.json', dict(count=9392, split_counts=list(map(len,splits)),
                   disjoint=True, matches_official_training=True, official_val_used_for_training=False))
        run = suite / 'full_train'
        run.mkdir()
        command = build_training_command(sys.executable, data, suite, 'full_train', 1e-4, 3e-4, epochs)
        command[command.index('--train-split')+1] = 'train+val+test'
        command[command.index('--eval-splits')+1] = ''
        update('training', active_trial='full_train', epochs=epochs, training_images=9392)
        execute('full_train', command, run / 'stdout.log')
        metrics = read_json(run / 'metrics.json')
        assert metrics['parameters'] == 125076960 and metrics['trainable_parameters'] == 39867360
        assert metrics['args']['train_split'] == 'train+val+test'
        assert metrics['args']['epochs'] == epochs and metrics['args']['schedule_epochs'] == epochs
        from run_bp7_schedule_search import sha256_file
        weights_hash = sha256_file(run / 'model.pt')
        output = Path(manifest['submission_directory']) / f'BP7_full9392_lr1x_cos{epochs}_e{epochs}_20261031_val_shift-0.2.zip'
        if output.exists():
            raise RuntimeError('Refusing to overwrite submission')
        update('generating_submission')
        execute('submission', [sys.executable, 'predict.py', '--run', str(run), '--weights', 'model.pt',
                '--images', str(official / 'val'), '--batch-size', '4', '--workers', '2',
                '--shift', '-0.2', '--out', str(output)], run / 'submission.log', single_gpu=True)
        update('checking_submission')
        execute('check_submission', [sys.executable, 'tools/check_submission.py', str(output),
                str(official / 'val/images.json')], run / 'submission_check.log', single_gpu=True)
        import zipfile, collections
        with zipfile.ZipFile(output) as z:
            assert z.testzip() is None
            detections = __import__('json').loads(z.read('detection.json'))
        images = read_json(official / 'val/images.json')['images']
        sizes = {im['id']: (im['width'], im['height']) for im in images}
        counts = collections.Counter(d['image_id'] for d in detections)
        assert len(images) == 1426 and set(counts) == set(sizes) and max(counts.values()) <= 100
        for d in detections:
            x, y, w, h = d['bbox']
            width, height = sizes[d['image_id']]
            assert x >= -1e-3 and y >= -1e-3 and x+w <= width+1e-3 and y+h <= height+1e-3
        assert output.stat().st_size < 25000000
        digest = sha256_file(output)
        output.with_suffix('.sha256').write_text(digest + '  ' + output.name + '\n')
        write_json(suite / 'submission_validation.json', dict(time=stamp(), path=str(output),
                   sha256=digest, checkpoint_sha256=weights_hash, epoch=epochs, schedule_epochs=epochs,
                   training_images=9392, prediction_images=1426, boxes=len(detections),
                   bytes=output.stat().st_size, format_passed=True, boxes_within_images=True,
                   shift=-0.2, uploaded=False))
        update('complete', active_trial=None, submission=str(output), checkpoint_sha256=weights_hash)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if dependency_lock is not None:
            dependency_lock.close()


if __name__ == '__main__':
    main()
