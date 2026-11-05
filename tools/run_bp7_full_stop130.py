"""One original BP7 full-data run: 160-epoch cosine, stop/export at epoch 130."""
import argparse
import collections
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from run_bp7_schedule_search import (
    build_training_command, configure_resources, read_json, sha256_file,
    stamp, verify_snapshot, write_json,
)


def training_command(python, data, suite, name='full_train'):
    command = build_training_command(python, data, suite, name, 1e-4, 3e-4, 160)
    command[command.index('--train-split') + 1] = 'train+val+test'
    command[command.index('--eval-splits') + 1] = ''
    return command + ['--stop-after-epoch', '130']


def resume_command(command, checkpoint):
    command = list(command)
    index = command.index('--stop-after-epoch')
    del command[index:index + 2]
    return command + ['--resume', str(checkpoint)]


def verify_stopped_run(run, command, epoch, images, *, expected_budget=(125076960, 39867360, 16)):
    """Validate the intentional early exit without expecting final metrics/model.pt."""
    import torch
    import train
    from predict import saved_args

    expected = vars(train.parse_args(command[command.index('train.py') + 1:]))
    expected = json.loads(json.dumps(expected, default=str))
    if saved_args(run) != expected:
        raise ValueError('Saved training arguments differ from the declared command')
    log_text = (run / 'log.txt').read_text()
    if f'Stopped cleanly after epoch {epoch};' not in log_text:
        raise ValueError('Training did not report a clean epoch-boundary stop')
    state = torch.load(run / 'training_state.pt', map_location='cpu', weights_only=False)
    signature = state['signature']
    steps = math.ceil(images / 16)
    if (state.get('format_version') != 1 or state['epoch'] != epoch
            or state['global_step'] != epoch * steps
            or state['scheduler']['last_epoch'] != epoch * steps):
        raise ValueError('Checkpoint epoch and scheduler progress disagree')
    if (signature['world_size'] != 2 or signature['dataset_size'] != images
            or signature['steps_per_epoch'] != steps or len(state['rank_rng']) != 2
            or state['ema'] is not None or state['history']):
        raise ValueError('Unexpected data coverage, ranks, EMA, or holdout evaluation')
    for key, value in signature['args'].items():
        if key not in expected or expected[key] != value:
            raise ValueError(f'Checkpoint configuration mismatch: {key}')
    for name, digest in signature['manifests'].items():
        if sha256_file(Path(expected['data']) / name) != digest:
            raise ValueError(f'Training manifest changed: {name}')
    parameters = signature['parameters']
    total = sum(math.prod(p['shape']) for p in parameters.values())
    trainable = sum(math.prod(p['shape']) for p in parameters.values() if p['trainable'])
    persistent_buffers = sum(t.numel() for k, t in state['model'].items()
                             if k not in parameters and t.is_floating_point())
    # DINOv3's 16 nonpersistent floating buffers do not appear in state_dict.
    budget = re.search(r'parameters: ([\d,]+) total, ([\d,]+) trainable '
                      r'\(\+ ([\d,]+) floating-point buffer values\)', log_text)
    if budget is None:
        raise ValueError('Missing runtime parameter/buffer budget report')
    reported = tuple(int(value.replace(',', '')) for value in budget.groups())
    buffers = reported[2]
    if (total, trainable, buffers) != tuple(expected_budget) or reported != (total, trainable, buffers):
        raise ValueError(f'Declared parameter budget mismatch: {total}, {trainable}, {buffers}')
    if total + buffers > 130000000 or trainable > 40000000:
        raise ValueError('Model exceeds the competition parameter budget')
    weights = run / f'ckpt_e{epoch}.pt'
    inference = torch.load(weights, map_location='cpu', weights_only=True)
    trained_names = {name for name, p in parameters.items() if p['trainable']}
    if not trained_names.issubset(inference):
        raise ValueError('Inference checkpoint lacks trained parameters')
    for name, tensor in inference.items():
        if name not in state['model'] or not torch.equal(tensor, state['model'][name]):
            raise ValueError(f'Inference and resumable checkpoint differ: {name}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite trained tensor: {name}')
    optimizer = state['optimizer']
    if not optimizer['state'] or any(int(s['step']) != epoch * steps for s in optimizer['state'].values()):
        raise ValueError('Optimizer state does not match the stopping step')
    factor = train.lr_factor(epoch * steps, round(expected['warmup_epochs'] * steps), 160 * steps)
    for group in optimizer['param_groups']:
        if group['initial_lr'] not in (1e-4, 3e-4) or not math.isclose(
                group['lr'], group['initial_lr'] * factor, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError('Learning rate does not follow the original 160-epoch cosine')
    return dict(time=stamp(), epoch=epoch, schedule_epochs=160, training_images=images,
                global_step=state['global_step'], steps_per_epoch=steps, parameters=total,
                trainable_parameters=trainable, buffer_values=buffers,
                persistent_buffer_values=persistent_buffers, lr_factor=factor,
                train_hours=state['train_hours'], training_gpu_hours=2 * state['train_hours'],
                checkpoint=str(weights), checkpoint_sha256=sha256_file(weights),
                training_state_sha256=sha256_file(run / 'training_state.pt'),
                trained_tensors_match=True, optimizer_and_rank_rng_saved=True)


def validate_zip(output, images_json):
    import csv
    import io
    from score_submission import check_format

    images = read_json(images_json)['images']
    sizes = {im['id']: (im['width'], im['height']) for im in images}
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or sorted(archive.namelist()) != ['classification.csv', 'detection.json']:
            raise ValueError('Corrupt or unexpected submission archive')
        text = archive.read('classification.csv').decode()
        if text.splitlines()[0].strip() != 'image_id,label':
            raise ValueError('Unexpected classification header')
        rows = list(csv.DictReader(io.StringIO(text)))
        detections = json.loads(archive.read('detection.json'))
    if len(rows) != len(images) or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate or missing classification rows')
    check_format(rows, detections, set(sizes))
    counts = collections.Counter(d['image_id'] for d in detections)
    if set(counts) != set(sizes) or max(counts.values()) > 100:
        raise ValueError('Unexpected detection image coverage or box cap')
    for detection in detections:
        x, y, w, h = detection['bbox']
        width, height = sizes[detection['image_id']]
        if not (x >= -1e-3 and y >= -1e-3 and x + w <= width + 1e-3 and y + h <= height + 1e-3):
            raise ValueError('Box outside the image')
    if output.stat().st_size >= 25000000:
        raise ValueError('Submission exceeds 25 MB')
    return dict(path=str(output), sha256=sha256_file(output), prediction_images=len(images),
                boxes=len(detections), bytes=output.stat().st_size, format_passed=True,
                boxes_within_images=True, shift=-0.2, uploaded=False)


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
                 maximum_new_training_runs=1, official_submissions=0)
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
