"""Run the fixed-BP7 local LR pair sequentially from a frozen source snapshot.

No official validation prediction or upload. The primary comparison is the final
130-epoch local-val score with the exact submission post-processing.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

from run_bp7_full130 import training_arguments


def stamp():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def training_args(data, multiplier):
    args = training_arguments(data)
    replacements = {
        '--train-split': 'train', '--eval-splits': 'val',
        '--lr': str(1e-4 * multiplier), '--backbone-lr': str(3e-4 * multiplier),
    }
    for flag, value in replacements.items():
        args[args.index(flag) + 1] = value
    return args + ['--eval-every', '10', '--eval-batch-size', '4',
                   '--warmup-epochs', '1', '--schedule-epochs', '130',
                   '--save-training-state', '--reproducible-loader']


def verify_smoke(full, resumed):
    import numpy as np
    import torch

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and np.array_equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    def compare_weights(filename):
        left = torch.load(full / filename, map_location='cpu', weights_only=True)
        right = torch.load(resumed / filename, map_location='cpu', weights_only=True)
        if set(left) != set(right):
            raise RuntimeError('Resumed model has different parameter keys')
        differences = [(left[k].float() - right[k].float()).abs().max().item()
                       for k in left if left[k].numel()]
        if not all(math.isfinite(d) for d in differences):
            raise RuntimeError('Nonfinite GPU smoke model parameters')
        return max(differences)

    before = compare_weights('ckpt_e1.pt')
    after = compare_weights('model.pt')
    states = [torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=False)
              for folder in (full, resumed)]
    for key in ('epoch', 'global_step', 'signature', 'scheduler', 'rank_rng'):
        if not equal(states[0][key], states[1][key]):
            raise RuntimeError(f'Uninterrupted/resumed training state differs: {key}')
    for folder, state in zip((full, resumed), states):
        if state['epoch'] != 2 or state['global_step'] != 4:
            raise RuntimeError('Smoke resumed to the wrong epoch/optimizer step')
        if not all(int(v['step']) == 4 for v in state['optimizer']['state'].values()):
            raise RuntimeError('AdamW step differs from the resumed global step')
        exported = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        if not all(torch.equal(value, state['model'][key]) for key, value in exported.items()):
            raise RuntimeError('Inference weights do not match saved training-state model')
    metrics = json.loads((resumed / 'metrics.json').read_text())
    if metrics['parameters'] != 125076960 or metrics['trainable_parameters'] != 39867360:
        raise RuntimeError('BP7 architecture changed')
    return {'maximum_parameter_difference_before_resume': before,
            'maximum_parameter_difference_after_resume': after,
            'rank_rng_equal': True, 'scheduler_equal': True, 'adam_steps': 4,
            'export_matches_training_state': True,
            'parameters': metrics['parameters'], 'trainable_parameters': metrics['trainable_parameters'],
            'state_epoch': states[1]['epoch'], 'state_global_step': states[1]['global_step'],
            'note': 'Independent GPU runs can differ numerically before restart (native ROIAlign backward '
                    'and cuDNN benchmark); CPU stochastic continuation is tested exactly. '
                    'This GPU check verifies state/RNG/steps and valid exports, not bitwise determinism.',
            'time': stamp()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    code = suite / 'code'
    lock = (suite / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite / 'status.json').exists():
        raise SystemExit('Suite already started; refusing to overwrite it')
    os.environ.update(
        CUDA_VISIBLE_DEVICES='0,1', NCCL_P2P_DISABLE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1',
        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
        HF_HOME='/mnt/hdd/junkaichen/hf-cache', HF_HUB_OFFLINE='1',
    )
    for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR',
                'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE']:
        os.environ.pop(key, None)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    state = {'pid': os.getpid(), 'started_at': stamp(), 'suite': str(suite),
             'gpus': [0, 1], 'cpu_affinity': sorted(os.sched_getaffinity(0)),
             'epochs_per_trial': 130, 'global_batch': 16, 'workers_per_rank': 2,
             'primary_endpoint': 'epoch130 local-val submission-post delta_m',
             'official_submissions': 0, 'completed_trials': []}

    def update(stage, **extra):
        state.update(stage=stage, updated_at=stamp(), **extra)
        write_json(suite / 'status.json', state)
        print(stamp(), stage, json.dumps(extra), flush=True)

    def execute(command, log_path, env=None):
        print(stamp(), shlex.join(command), flush=True)
        with log_path.open('a') as output:
            with subprocess.Popen(command, cwd=code, env=env, stdout=output,
                                  stderr=subprocess.STDOUT) as child:
                state['child_pid'] = child.pid
                write_json(suite / 'status.json', state)
                returncode = child.wait()
        state.pop('child_pid', None)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)

    ddp = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node', '2', 'train.py']
    validation = suite / 'validation'
    commands = {}
    try:
        update('verifying_inputs')
        for line in (suite / 'source.sha256').read_text().splitlines():
            expected, name = line.split('  ', 1)
            if hashlib.sha256((code / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Source checksum mismatch: {name}')
        manifest = json.loads((suite / 'launch_manifest.json').read_text())
        for name, expected in manifest['data_sha256'].items():
            if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Data checksum mismatch: {name}')
        validation.mkdir()
        smoke = ddp + training_args(args.data, 1.0) + [
            '--out', str(validation), '--epochs', '2', '--schedule-epochs', '2',
            '--limit', '24', '--eval-every', '1', '--save-every', '1']
        full, resumed = validation / 'full', validation / 'resumed'
        commands['smoke_full'] = smoke + ['--name', 'full']
        commands['smoke_stop'] = smoke + ['--name', 'resumed', '--stop-after-epoch', '1']
        commands['smoke_resume'] = smoke + ['--name', 'resumed', '--resume', str(resumed / 'training_state.pt')]
        for name, multiplier in [('lr1x', 1.0), ('lr05x', 0.5)]:
            run = suite / name
            commands[name] = ddp + training_args(args.data, multiplier) + [
                '--out', str(suite), '--name', name]
            commands[name + '_evaluate'] = [
                sys.executable, 'tools/evaluate_lr_checkpoint.py', '--run', str(run),
                '--data', str(args.data), '--split', 'val', '--weights', 'model.pt',
                '--reference', str(args.reference), '--batch-size', '4', '--workers', '2',
                '--out', str(run / 'metrics_submission_post_val.json')]
        commands['smoke_evaluate'] = [
            sys.executable, 'tools/evaluate_lr_checkpoint.py', '--run', str(full),
            '--data', str(args.data), '--split', 'val', '--reference', str(args.reference),
            '--batch-size', '4', '--workers', '2',
            '--out', str(validation / 'evaluation_smoke.json')]
        write_json(suite / 'commands.json', commands)
        for label in ['smoke_full', 'smoke_stop', 'smoke_resume']:
            update(label)
            execute(commands[label], validation / f'{label}.log')
        smoke_result = verify_smoke(full, resumed)
        write_json(validation / 'resume_smoke_passed.json', smoke_result)
        single = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
        update('smoke_submission_post')
        execute(commands['smoke_evaluate'], validation / 'smoke_evaluate.log', single)
        for name in ['lr1x', 'lr05x']:
            run = suite / name
            run.mkdir()
            update('training', active_trial=name)
            execute(commands[name], run / 'stdout.log')
            update('evaluating_submission_post', active_trial=name)
            execute(commands[name + '_evaluate'], run / 'evaluation.log', single)
            state['completed_trials'].append(name)
            update('trial_complete', active_trial=name)
        results = {name: json.loads((suite / name / 'metrics_submission_post_val.json').read_text())
                   for name in ['lr1x', 'lr05x']}
        write_json(suite / 'comparison.json', {
            'time': stamp(), 'selection_metric': state['primary_endpoint'],
            'half_minus_baseline_delta_m': results['lr05x']['local_delta_m'] - results['lr1x']['local_delta_m'],
            'higher_scoring_trial': max(results, key=lambda name: results[name]['local_delta_m']),
            'results': results, 'caution': 'One seed; local scores are not official leaderboard scores.',
            'test_evaluated': False, 'automatic_full_data_retrain': False,
        })
        update('complete', active_trial=None)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
