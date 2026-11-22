"""Continue BP7 e147 to e160 with private temporary files and a progress watchdog."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from run_bp7_cos200_final import configure_resources, verify_data, verify_stopped_run, validate_zip
from run_bp7_e130_recovery import continuation_command, value, verify_frozen
from run_bp7_schedule_search import read_json, sha256_file, stamp, write_json

ALLOCATOR = 'expandable_segments:True'
RESOURCE_ERRORS = ('No space left on device', 'OSError: [Errno 28]', 'CUDA out of memory',
                   'DataLoader worker', 'Bus error', 'Too many open files')
PROGRESS = re.compile(r'epoch (\d+) iter (\d+)/(\d+)')
IGNORED_ARGS = {'out', 'name', 'resume', 'stop_after_epoch', 'save_training_state',
                'save', 'save_every', 'dump_raw', 'allow_over_budget'}


def configure_temp(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or len(os.fsencode(path)) + 32 >= 108:
        raise ValueError('Need an absolute, short, non-symlink temporary directory')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    usage = os.statvfs(path)
    if usage.f_bavail * usage.f_frsize < 1024 ** 3 or usage.f_favail < 100:
        raise ValueError('Insufficient temporary filesystem space or inodes')
    os.environ.update(TMPDIR=str(path), TMP=str(path), TEMP=str(path))
    tempfile.tempdir = str(path)
    with tempfile.TemporaryDirectory(prefix='check-', dir=path) as check:
        Path(check, 'write').write_bytes(b'ok')
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(Path(check, 'listener')))
    return dict(path=str(path), free_bytes=usage.f_bavail * usage.f_frsize, free_inodes=usage.f_favail)


def verify_state(checkpoint, command, epoch, expected_sha=None, *, budget=(125076960, 39867360)):
    if expected_sha is not None and sha256_file(checkpoint) != expected_sha:
        raise ValueError('Resume checkpoint checksum mismatch')
    import torch
    import train
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if epoch is None:
        epoch = state['epoch']
        if not isinstance(epoch, int) or not 148 <= epoch <= 160:
            raise ValueError('First resumed checkpoint is outside epochs 148 through 160')
    signature, steps = state['signature'], epoch * 587
    if (state.get('format_version') != 1 or state['epoch'] != epoch or state['global_step'] != steps
            or state['scheduler']['last_epoch'] != steps or signature['world_size'] != 2
            or signature['dataset_size'] != 9392 or signature['steps_per_epoch'] != 587
            or len(state['rank_rng']) != 2 or state['ema'] is not None or state['history']):
        raise ValueError('Resume state has incorrect epoch, updates, dataset, ranks or evaluation')
    for rng in state['rank_rng']:
        if set(rng) != {'python', 'numpy', 'torch', 'cuda'} or rng['cuda'] is None:
            raise ValueError('Incomplete per-rank RNG state')
    args = train.parse_args(command[command.index('train.py') + 1:])
    config = json.loads(json.dumps({k: v for k, v in vars(args).items() if k not in IGNORED_ARGS}, default=str))
    if signature['args'] != config:
        raise ValueError('Resume training configuration changed')
    if args.epochs != 200 or args.schedule_epochs != 200 or args.batch_size != 16 or args.workers != 1:
        raise ValueError('Unexpected BP7 training recipe')
    for name, digest in signature['manifests'].items():
        if sha256_file(args.data / name) != digest:
            raise ValueError(f'Training manifest changed: {name}')
    parameters = signature['parameters']
    counts = (sum(math.prod(p['shape']) for p in parameters.values()),
              sum(math.prod(p['shape']) for p in parameters.values() if p['trainable']))
    if counts != budget:
        raise ValueError('Model parameter budget changed')
    for name, spec in parameters.items():
        tensor = state['model'].get(name)
        if tensor is None or list(tensor.shape) != spec['shape'] or not torch.isfinite(tensor).all():
            raise ValueError(f'Missing, malformed or nonfinite model tensor: {name}')
    optimizer = state['optimizer']
    if not optimizer['state'] or any(int(s['step']) != steps for s in optimizer['state'].values()):
        raise ValueError('Optimizer progress differs from the resume epoch')
    factor = train.lr_factor(steps, round(args.warmup_epochs * 587), 200 * 587)
    for group in optimizer['param_groups']:
        if group['initial_lr'] not in (1e-4, 3e-4) or not math.isclose(
                group['lr'], group['initial_lr'] * factor, rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError('Resume learning rate differs from the original cosine schedule')
    return dict(time=stamp(), epoch=epoch, global_step=steps, checkpoint=str(checkpoint),
                sha256=sha256_file(checkpoint), lr_factor=factor, train_hours=state['train_hours'],
                optimizer_and_rank_rng_saved=True)


def verify_inputs(suite, manifest):
    source = Path(manifest['source_suite']).resolve()
    if suite == source or suite.is_relative_to(source) or manifest['source_epoch'] != 147:
        raise ValueError('Need a separate recovery suite and the e147 checkpoint')
    previous, current = verify_frozen(source), verify_frozen(suite)
    if any(current.get(name) != digest for name, digest in previous.items()):
        raise ValueError('Original frozen implementation changed')
    if 'tools/run_bp7_e147_recovery.py' not in current:
        raise ValueError('Recovery driver missing from frozen manifest')
    if read_json(source / 'status.json')['stage'] != 'failed':
        raise ValueError('Source must be stopped and failed')
    if manifest['gpu_memory_fraction'] != 0.35 or manifest['allocator_config'] != ALLOCATOR:
        raise ValueError('Unexpected GPU settings')
    original = read_json(source / 'launch_manifest.json')
    for key in ('data', 'official_data'):
        if Path(original[key]).resolve() != Path(manifest[key]).resolve():
            raise ValueError(f'Recovery {key} changed')
    command = read_json(source / 'commands.json')['full_train_e160']
    for flag, expected in {'--memory-fraction': '0.35', '--stop-after-epoch': '160',
                           '--save-every': '10', '--out': str(source), '--name': 'full_train_e160'}.items():
        if value(command, flag) != expected:
            raise ValueError(f'Unexpected source command: {flag}')
    checkpoint = source / 'full_train_e160/training_state.pt'
    checked = verify_state(checkpoint, command, 147, manifest['source_checkpoint_sha256'])
    existing, output = Path(manifest['existing_e130_submission']), Path(manifest['submission_path'])
    if not output.is_absolute() or output.resolve() == existing.resolve():
        raise ValueError('Need a distinct absolute e160 output')
    if output.exists() or output.with_suffix('.sha256').exists():
        raise FileExistsError('Refusing to overwrite an existing submission')
    if sha256_file(existing) != manifest['existing_e130_submission_sha256']:
        raise ValueError('Original e130 submission changed')
    checked['existing_e130_submission'] = validate_zip(existing, Path(manifest['official_data']) / 'val/images.json')
    return command, checked


def owned_command(argv, suite):
    for flag in ('--out', '--run'):
        if flag in argv:
            position = argv.index(flag) + 1
            if position < len(argv) and Path(argv[position]) in (suite, suite / 'full_train_e160'):
                return True
    return False


def process_table():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = [p.decode(errors='replace') for p in (path / 'cmdline').read_bytes().split(b'\0') if p]
            if stat[0] != 'Z':
                result[int(path.name)] = dict(group=int(stat[2]), start=stat[19], argv=argv)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def kill_remaining_owned(suite):
    table = process_table()
    owned = {pid: row for pid, row in table.items() if owned_command(row['argv'], suite)}
    signaled_groups = set()
    for pid, row in owned.items():
        now_table = process_table()
        now = now_table.get(pid)
        if now is None or now['start'] != row['start'] or not owned_command(now['argv'], suite):
            continue
        group = now['group']
        group = now['group']
        try:
            members = {p for p, r in now_table.items() if r['group'] == group}
            if group != os.getpgrp() and all(owned_command(now_table[p]['argv'], suite) for p in members):
                if group not in signaled_groups:
                    os.killpg(group, signal.SIGKILL)
                    signaled_groups.add(group)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_child(child, suite):
    if child.poll() is None:
        child.terminate()  # Let torchrun propagate SIGTERM first.
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    kill_remaining_owned(suite)  # Ranks can have independent process groups.
    child.wait(timeout=10)
    for _ in range(50):
        if not any(owned_command(p['argv'], suite) for p in process_table().values()):
            return
        time.sleep(0.1)
    raise RuntimeError('Owned training processes remain after watchdog cleanup')


def check_progress(text, previous, last_progress, now, training):
    if any(error in text for error in RESOURCE_ERRORS):
        raise RuntimeError('Resource error in child log; stopping without retry')
    matches = [(int(a), int(b)) for a, b, _ in PROGRESS.findall(text)]
    current = max([previous, *matches])
    if current > previous:
        last_progress = now
    if training and now - last_progress >= 300:
        raise TimeoutError('Training has made no epoch/iteration progress for 300 seconds')
    return current, last_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    suite = parser.parse_args().suite.resolve()
    if Path(__file__).resolve().parents[1] != suite / 'code':
        raise RuntimeError('Run only from the frozen source snapshot')
    with (suite / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (suite / 'status.json').exists():
            raise FileExistsError('Refusing to overwrite a previous recovery attempt')
        manifest = read_json(suite / 'launch_manifest.json')
        temporary = configure_temp(manifest['temp_dir'])
        configure_resources(ALLOCATOR)
        state = dict(pid=os.getpid(), started_at=stamp(), suite=str(suite), gpu_memory_fraction=0.35,
                     allocator_config=ALLOCATOR, temp_dir=manifest['temp_dir'], gpus=[0, 1],
                     global_batch=16, schedule_epochs=200, stop_after_epoch=160,
                     official_submissions=0, submissions={'130': manifest['existing_e130_submission']})
        commands = {}
        def update(stage=None, **extra):
            if stage:
                state['stage'] = stage
            state.update(updated_at=stamp(), **extra)
            write_json(suite / 'status.json', state)
            if stage:
                print(stamp(), stage, extra, flush=True)
        def execute(name, command, log, training=False):
            commands[name] = command
            write_json(suite / 'commands.json', commands)
            env = dict(os.environ)
            if not training:
                env['CUDA_VISIBLE_DEVICES'] = '1'
            progress = (147, 587) if training else (state.get('epoch', 160), state.get('iteration', 587))
            last_progress = time.monotonic()
            observed = False
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=suite / 'code', env=env, start_new_session=True,
                                         stdout=stream, stderr=subprocess.STDOUT)
                try:
                    update(child_pid=child.pid)
                    while True:
                        try:
                            rc = child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            rc = None
                        progress, last_progress = check_progress(log.read_text(errors='replace'), progress,
                                                                 last_progress, time.monotonic(), training)
                        if training and not observed and (log.parent / 'training_state.pt').exists():
                            checked = verify_state(log.parent / 'training_state.pt', command, None)
                            write_json(suite / 'checkpoint_after_resume_validation.json', checked)
                            observed = True
                        update(epoch=progress[0], iteration=progress[1],
                               seconds_since_progress=round(time.monotonic() - last_progress, 1))
                        if rc is not None:
                            if rc:
                                raise subprocess.CalledProcessError(rc, command)
                            break
                except BaseException:
                    stop_child(child, suite)
                    raise
                finally:
                    state.pop('child_pid', None)
        try:
            update('verifying_inputs')
            write_json(suite / 'temporary_directory_check.json', temporary)
            source = Path(manifest['source_suite']).resolve()
            with (source / 'driver.lock').open('r') as source_lock:
                fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                template, checked = verify_inputs(suite, manifest)
                write_json(suite / 'input_verification.json', checked)
                _, data_check = verify_data(Path(manifest['data']), Path(manifest['official_data']))
                write_json(suite / 'full_data_check.json', data_check)
                run = suite / 'full_train_e160'
                run.mkdir()
                command = continuation_command(template, suite, 'full_train_e160', 160,
                                               source / 'full_train_e160/training_state.pt', 10)
                update('training', target_epoch=160, resumed_epoch=147)
                execute('full_train_e160', command, run / 'stdout.log', training=True)
                update('verifying_epoch160')
                result = verify_stopped_run(run, command, 160, 9392)
                write_json(suite / 'checkpoint_e160_validation.json', result)
                output, images = Path(manifest['submission_path']), Path(manifest['official_data']) / 'val/images.json'
                if output.exists() or output.with_suffix('.sha256').exists():
                    raise FileExistsError('Refusing to overwrite submission output')
                prediction = [sys.executable, 'tools/train_gpu_limited.py', '--memory-fraction', '0.35',
                    'predict.py', '--run', str(run), '--weights', 'ckpt_e160.pt',
                    '--images', str(images.parent), '--images-json', str(images),
                    '--batch-size', '2', '--workers', '0', '--shift', '-0.2', '--out', str(output)]
                update('generating_submission_e160')
                execute('submission_e160', prediction, run / 'submission.log')
                exported = validate_zip(output, images)
                exported.update(epoch=160, checkpoint_sha256=result['checkpoint_sha256'])
                write_json(suite / 'submission_e160_validation.json', exported)
                with output.with_suffix('.sha256').open('x') as stream:
                    stream.write(exported['sha256'] + '  ' + output.name + '\n')
                state['submissions']['160'] = str(output)
                update('complete', train_hours=result['train_hours'])
        except Exception as error:
            update('failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
