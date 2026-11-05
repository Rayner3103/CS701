"""Finish BP7 cosine-200 from e160, preserving e180 and final e200 exports."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_bp7_cos200_final import configure_resources, verify_data, validate_zip
from run_bp7_e130_recovery import MUTABLE_FLAGS, value, verify_frozen
from run_bp7_e147_recovery import (ALLOCATOR, configure_temp, check_progress, process_table,
                                  verify_state)
from run_bp7_schedule_search import read_json, sha256_file, stamp, write_json

RUN_NAME = 'full_train_e200'


def continuation_command(template, suite, checkpoint):
    """Only output/resume/stopping bookkeeping changes; all training args persist."""
    command = list(template)
    for flag, replacement in zip(MUTABLE_FLAGS, (0.35, suite, RUN_NAME, 200, 10, checkpoint)):
        value(command, flag)
        command[command.index(flag) + 1] = str(replacement)
    return command


def output_paths(manifest):
    outputs = {int(epoch): Path(path) for epoch, path in manifest['submission_paths'].items()}
    paths = [Path(manifest['existing_e160_submission']), *outputs.values()]
    if (set(outputs) != {180, 200} or any(not p.is_absolute() for p in paths)
            or len({p.resolve() for p in paths}) != 3):
        raise ValueError('Need distinct absolute protected-e160, e180 and e200 paths')
    for path in outputs.values():
        if path.exists() or path.with_suffix('.sha256').exists():
            raise FileExistsError('Refusing to overwrite an existing submission')
    return outputs


def verify_inputs(suite, manifest):
    source = Path(manifest['source_suite']).resolve()
    if (suite == source or suite.is_relative_to(source) or source.is_relative_to(suite)
            or manifest['source_epoch'] != 160):
        raise ValueError('Need a separate continuation suite and the e160 checkpoint')
    previous, current = verify_frozen(source), verify_frozen(suite)
    if any(current.get(name) != digest for name, digest in previous.items()):
        raise ValueError('Original frozen implementation changed')
    if 'tools/run_bp7_e160_to200.py' not in current:
        raise ValueError('Continuation driver missing from frozen manifest')
    if read_json(source / 'status.json')['stage'] != 'complete':
        raise ValueError('Source must have completed e160')
    if manifest['gpu_memory_fraction'] != 0.35 or manifest['allocator_config'] != ALLOCATOR:
        raise ValueError('Unexpected GPU settings')
    original = read_json(source / 'launch_manifest.json')
    for key in ('data', 'official_data'):
        if (not Path(manifest[key]).is_absolute()
                or Path(original[key]).resolve() != Path(manifest[key]).resolve()):
            raise ValueError(f'Continuation {key} changed')
    command = read_json(source / 'commands.json')['full_train_e160']
    expected = {'--memory-fraction': '0.35', '--stop-after-epoch': '160', '--save-every': '10',
                '--out': str(source), '--name': 'full_train_e160',
                '--train-split': 'train+val+test', '--eval-splits': ''}
    for flag, wanted in expected.items():
        if value(command, flag) != wanted:
            raise ValueError(f'Unexpected source command: {flag}')
    import train
    effective = train.parse_args(command[command.index('train.py') + 1:])
    if effective.save != 'trainable':
        raise ValueError('Source must save trainable inference weights')
    if (command[:6] != [command[0], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2']
            or not Path(command[0]).is_absolute() or '--compile' in command
            or not all(f in command for f in ('--grad-ckpt', '--reproducible-loader', '--save-training-state'))):
        raise ValueError('Unexpected distributed training command')
    checkpoint = source / 'full_train_e160/training_state.pt'
    checked = verify_state(checkpoint, command, 160, manifest['source_checkpoint_sha256'])
    output_paths(manifest)
    existing = Path(manifest['existing_e160_submission'])
    if existing.resolve() != Path(original['submission_path']).resolve():
        raise ValueError('Protected e160 ZIP is not the completed source export')
    if sha256_file(existing) != manifest['existing_e160_submission_sha256']:
        raise ValueError('Original e160 submission changed')
    checked['existing_e160_submission'] = validate_zip(existing, Path(manifest['official_data']) / 'val/images.json')
    return command, checked


def verify_inference(weights, state):
    """Every trained tensor and persistent buffer must match the same full state."""
    import torch
    parameters = state['signature']['parameters']
    expected = {n for n, p in parameters.items() if p['trainable']} | (set(state['model']) - set(parameters))
    inference = torch.load(weights, map_location='cpu', weights_only=True)
    if set(inference) != expected:
        raise ValueError('Inference checkpoint lacks trained tensors or persistent buffers')
    for name, tensor in inference.items():
        if not torch.equal(tensor, state['model'][name]) or not torch.isfinite(tensor).all():
            raise ValueError(f'Inference and resumable weights differ or are nonfinite: {name}')
    return dict(checkpoint=str(weights), checkpoint_sha256=sha256_file(weights),
                trained_tensors_match=True, persistent_buffers_match=True)


def observe_checkpoint(run, command, observed, *, budget=(125076960, 39867360)):
    """Pin an atomic checkpoint inode while verifying; never pause GPU training."""
    path = run / 'training_state.pt'
    if (not path.exists() or (observed.get('first') and (observed.get('180') or observed.get('180_skipped')))
            or (observed.get('first') and not (run / 'ckpt_e180.pt').exists())):
        return {}
    import torch
    checks = {}
    # Atomic os.replace may publish the next epoch while CPU validation runs.
    # Opening once and using /proc/self/fd pins the exact same old inode for all reads.
    with path.open('rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
        epoch = state['epoch']
        if not isinstance(epoch, int) or not 161 <= epoch <= 200:
            raise ValueError('Resumed state is outside epochs 161 through 200')
        if not observed.get('first') or epoch == 180:
            checked = verify_state(Path('/proc/self/fd') / str(stream.fileno()), command, epoch, budget=budget)
            checked['checkpoint'] = str(path)
            if not observed.get('first'):
                checks['first'] = dict(checked)
            if epoch == 180:
                checked.update(verify_inference(run / 'ckpt_e180.pt', state))
                checks['180'] = checked
        if epoch > 180 and not observed.get('180'):
            checks['180_skipped'] = dict(time=stamp(), observed_epoch=epoch,
                reason='Controller missed the e180 full-state verification window; e200 export remains independent')
    return checks


def verify_final_run(run, command, *, budget=(125076960, 39867360, 16)):
    """Natural epoch 200 writes model.pt/metrics.json, not ckpt_e200.pt."""
    import torch
    import train
    expected = json.loads(json.dumps(vars(train.parse_args(command[command.index('train.py') + 1:])), default=str))
    if (expected['epochs'], expected['schedule_epochs'], expected['stop_after_epoch']) != (200, 200, 200):
        raise ValueError('Expected natural completion of the unchanged cosine-200 run')
    metrics = read_json(run / 'metrics.json')
    if metrics['args'] != expected or metrics['val_history'] or expected['eval_splits']:
        raise ValueError('Final metrics arguments or evaluation differ from the command')
    reported = (metrics['parameters'], metrics['trainable_parameters'], metrics['buffer_values'])
    if reported != budget or reported[0] + reported[2] > 130000000 or reported[1] > 40000000:
        raise ValueError('Final parameter budget mismatch')
    log = (run / 'log.txt').read_text()
    if 'epoch 200 iter 587/587' not in log or 'RESULT {}' not in log:
        raise ValueError('Final training completion is absent from the log')
    checked = verify_state(run / 'training_state.pt', command, 200, budget=budget[:2])
    state = torch.load(run / 'training_state.pt', map_location='cpu', weights_only=False)
    if not math.isclose(metrics['train_hours'], state['train_hours'], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Final metrics and state training duration differ')
    checked.update(verify_inference(run / 'model.pt', state))
    checked.update(parameters=reported[0], trainable_parameters=reported[1], buffer_values=reported[2],
                   schedule_epochs=200, training_images=9392, training_state_sha256=checked['sha256'],
                   final_metrics=str(run / 'metrics.json'))
    return checked


def owned_command(argv, suite):
    for flag in ('--out', '--run'):
        if flag in argv:
            i = argv.index(flag) + 1
            if i < len(argv) and Path(argv[i]) in (suite, suite / RUN_NAME):
                return True
    return False


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
        child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    kill_remaining_owned(suite)
    child.wait(timeout=10)
    for _ in range(50):
        if not any(owned_command(p['argv'], suite) for p in process_table().values()):
            return
        time.sleep(0.1)
    raise RuntimeError('Owned training processes remain after watchdog cleanup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    suite = parser.parse_args().suite.resolve()
    if Path(__file__).resolve().parents[1] != suite / 'code':
        raise RuntimeError('Run only from the frozen source snapshot')
    with (suite / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (suite / 'status.json').exists():
            raise FileExistsError('Refusing to overwrite a previous continuation attempt')
        manifest = read_json(suite / 'launch_manifest.json')
        temporary = configure_temp(manifest['temp_dir'])
        configure_resources(ALLOCATOR)
        state = dict(pid=os.getpid(), started_at=stamp(), suite=str(suite), gpu_memory_fraction=0.35,
                     allocator_config=ALLOCATOR, temp_dir=manifest['temp_dir'], gpus=[0, 1],
                     global_batch=16, schedule_epochs=200, stop_after_epoch=200,
                     official_submissions=0, submissions={'160': manifest['existing_e160_submission']})
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
            progress = (160, 587) if training else (state.get('epoch', 200), state.get('iteration', 587))
            last_progress, observed = time.monotonic(), {}
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
                        if training:
                            checks = observe_checkpoint(log.parent, command, observed)
                            for key, checked in checks.items():
                                filename = 'checkpoint_after_resume_validation.json' if key == 'first' else f'checkpoint_e{key}_validation.json'
                                write_json(suite / filename, checked)
                            observed.update(checks)
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
            return observed

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
                run = suite / RUN_NAME
                run.mkdir()
                command = continuation_command(template, suite, source / 'full_train_e160/training_state.pt')
                update('training', target_epoch=200, resumed_epoch=160)
                observed = execute(RUN_NAME, command, run / 'stdout.log', training=True)
                update('verifying_epoch200')
                result = verify_final_run(run, command)
                write_json(suite / 'checkpoint_e200_validation.json', result)
                exports = [(200, 'model.pt', result)]
                midpoint = observed.get('180')
                if midpoint and midpoint['checkpoint_sha256'] == sha256_file(run / 'ckpt_e180.pt'):
                    exports.append((180, 'ckpt_e180.pt', midpoint))
                else:
                    update(skipped_submission_e180='The e180 checkpoint was not verified against its exact full state')
                images = Path(manifest['official_data']) / 'val/images.json'
                outputs = output_paths(manifest)
                # Finish all training before either export. e200 remains independent of e180.
                for epoch, weights, verified in exports:
                    output = outputs[epoch]
                    if output.exists() or output.with_suffix('.sha256').exists():
                        raise FileExistsError('Refusing to overwrite a submission output')
                    output.parent.mkdir(parents=True, exist_ok=True)
                    prediction = [sys.executable, 'tools/train_gpu_limited.py', '--memory-fraction', '0.35',
                        'predict.py', '--run', str(run), '--weights', weights,
                        '--images', str(images.parent), '--images-json', str(images),
                        '--batch-size', '2', '--workers', '0', '--shift', '-0.2', '--out', str(output)]
                    update(f'generating_submission_e{epoch}')
                    execute(f'submission_e{epoch}', prediction, run / f'submission_e{epoch}.log')
                    exported = validate_zip(output, images)
                    exported.update(epoch=epoch, checkpoint_sha256=verified['checkpoint_sha256'])
                    write_json(suite / f'submission_e{epoch}_validation.json', exported)
                    with output.with_suffix('.sha256').open('x') as stream:
                        stream.write(exported['sha256'] + '  ' + output.name + '\n')
                    state['submissions'][str(epoch)] = str(output)
                update('complete', train_hours=result['train_hours'])
        except Exception as error:
            update('failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
