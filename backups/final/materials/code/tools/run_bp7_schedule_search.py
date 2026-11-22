"""Bounded BP7 follow-up: one grouped-LR trial, then one 160-epoch trial.

Wait for the existing pair without touching it. Select on local validation only;
local test confirms the already fixed candidate. No upload or full-data retrain.
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
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from run_bp7_lr_trials import training_args

LR_PREFERENCE = ('lr1x', 'hybrid130', 'lr05x')
MARGIN = 1.0  # Operational indifference band, NOT a statistical significance threshold.
EXPECTED_PARAMETERS = 125076960
EXPECTED_TRAINABLE = 39867360
POLL_SECONDS = 30
WAIT_SECONDS = 48 * 3600


def stamp():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f'Nonfinite JSON number in {path}')
        return number

    def invalid_constant(value):
        raise ValueError(f'Invalid JSON constant {value} in {path}')

    return json.loads(Path(path).read_text(), parse_float=finite_float, parse_constant=invalid_constant)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected a finite score, got {value!r}')
    return float(value)


def choose_lr_candidate(scores, margin=MARGIN):
    if set(scores) != set(LR_PREFERENCE):
        raise ValueError(f'Expected exactly {LR_PREFERENCE}')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('The indifference margin must be finite and nonnegative')
    values = {name: finite_score(value) for name, value in scores.items()}
    best = max(values.values())
    return next(name for name in LR_PREFERENCE if best - values[name] <= margin)


def choose_duration(short_name, short_score, long_score, margin=MARGIN):
    if short_name not in LR_PREFERENCE:
        raise ValueError(f'Unknown 130-epoch candidate: {short_name}')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('The indifference margin must be finite and nonnegative')
    improvement = finite_score(long_score) - finite_score(short_score)
    return 'schedule160' if improvement > margin else short_name


def _training_arguments(data, suite, name, head_lr, backbone_lr, epochs):
    head_lr, backbone_lr = finite_score(head_lr), finite_score(backbone_lr)
    if epochs not in (130, 160) or head_lr <= 0 or backbone_lr <= 0:
        raise ValueError('Only the predeclared 130/160-epoch positive-LR trials are allowed')
    args = training_args(Path(data).resolve(), 1.0)
    for flag, value in {'--lr': head_lr, '--backbone-lr': backbone_lr,
                        '--epochs': epochs, '--schedule-epochs': epochs}.items():
        args[args.index(flag) + 1] = str(value)
    return args + ['--out', str(Path(suite).resolve()), '--name', name]


def build_training_command(python, data, suite, name, head_lr, backbone_lr, epochs):
    return [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2', 'train.py',
            *_training_arguments(data, suite, name, head_lr, backbone_lr, epochs)]


def dependency_ready(state, locked, alive):
    """`locked` means the dependency lock is held by another process."""
    stage = state.get('stage')
    if stage == 'failed':
        raise RuntimeError(f'Dependency failed: {state.get("error", "unknown error")}')
    if stage == 'complete':
        return not locked
    if not alive:
        raise RuntimeError(f'Dependency driver died before completion (stage={stage!r})')
    return False


def process_is_dependency(pid, suite):
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        if not command or not any(b'run_bp7_lr_trials.py' in arg for arg in command):
            return False
        return str(Path(suite).resolve()).encode() in command
    except (ProcessLookupError, FileNotFoundError):
        return False


def verify_snapshot(suite, dependency, data, reference=None):
    suite, dependency, data = map(Path, (suite, dependency, data))
    code, previous = suite / 'code', dependency / 'code'
    entries = (suite / 'source.sha256').read_text().splitlines()
    if not entries:
        raise ValueError('Empty source checksum manifest')
    for line in entries:
        expected, name = line.split('  ', 1)
        source = (code / name).resolve()
        if not source.is_relative_to(code.resolve()) or sha256_file(source) != expected:
            raise ValueError(f'Frozen source checksum mismatch: {name}')
    manifest = read_json(suite / 'launch_manifest.json')
    earlier = read_json(dependency / 'launch_manifest.json')
    if manifest.get('policy_margin') != MARGIN:
        raise ValueError('The frozen policy margin differs from this driver')
    if manifest.get('expected_parameters') != EXPECTED_PARAMETERS or manifest.get('expected_trainable_parameters') != EXPECTED_TRAINABLE:
        raise ValueError('The frozen architecture budget differs from BP7')
    if reference is not None and manifest.get('reference_sha256') != sha256_file(reference):
        raise ValueError('The local reference differs from the frozen manifest')
    if manifest.get('data_sha256') != earlier.get('data_sha256') or not manifest.get('data_sha256'):
        raise ValueError('New and existing trials do not use the same data manifests')
    for name, expected in manifest['data_sha256'].items():
        source = (data / name).resolve()
        if not source.is_relative_to(data.resolve()) or sha256_file(source) != expected:
            raise ValueError(f'Data checksum mismatch: {name}')
    shared = {'train.py', 'predict.py', 'pp_eval.py', 'pp_study.py', 'make_submission.py',
              'tools/evaluate_lr_checkpoint.py', 'tools/run_bp7_lr_trials.py', 'tools/run_bp7_full130.py'}
    prior_modules = {str(path.relative_to(previous)) for path in (previous / 'sarbench').rglob('*.py')}
    current_modules = {str(path.relative_to(code)) for path in (code / 'sarbench').rglob('*.py')}
    if prior_modules != current_modules or not prior_modules:
        raise ValueError('The sarbench module sets differ between snapshots')
    for name in sorted(shared | prior_modules):
        if sha256_file(code / name) != sha256_file(previous / name):
            raise ValueError(f'Training/evaluation implementation changed: {name}')
    return {'source_files': len(entries), 'matched_implementation_files': len(shared | prior_modules),
            'data_manifests': len(manifest['data_sha256'])}


def validate_result(run, path, *, data, reference, epochs, head_lr, backbone_lr, split='val'):
    """Accept only a complete, matching, uncontaminated local result with authentic weights."""
    import train
    from evaluate_lr_checkpoint import local_delta_m, postprocessing_arguments

    run, data, reference = Path(run).resolve(), Path(data).resolve(), Path(reference).resolve()
    if split not in ('val', 'test'):
        raise ValueError('Only the predeclared local holdouts are allowed')
    result = read_json(path)
    metrics = read_json(run / 'metrics.json')
    expected_args = vars(train.parse_args(_training_arguments(data, run.parent, run.name,
                                                            head_lr, backbone_lr, epochs)))
    expected_args = json.loads(json.dumps(expected_args, default=str))
    if metrics.get('args') != expected_args or result.get('training_args') != expected_args:
        raise ValueError(f'Training configuration differs from the declared trial: {run}')
    if metrics.get('parameters') != EXPECTED_PARAMETERS or metrics.get('trainable_parameters') != EXPECTED_TRAINABLE:
        raise ValueError(f'BP7 parameter budget mismatch: {run}')
    if metrics['parameters'] + metrics.get('buffer_values', 0) > 130000000 or metrics['trainable_parameters'] > 40000000:
        raise ValueError(f'The model exceeds the parameter budget: {run}')
    if result.get('split') != split or result.get('n_images') != 940 or result.get('training_splits') != ['train']:
        raise ValueError(f'Unexpected evaluation split or coverage: {path}')
    if Path(result['data']).resolve() != data or Path(result['run']).resolve() != run:
        raise ValueError(f'Unexpected evaluation source: {path}')
    if result.get('post') != vars(postprocessing_arguments()):
        raise ValueError(f'Submission post-processing mismatch: {path}')
    weights = run / 'model.pt'
    if Path(result['weights']).resolve() != weights or result.get('weights_sha256') != sha256_file(weights):
        raise ValueError(f'Weight checksum mismatch: {path}')
    ground_truth = data / 'detection' / f'instances_{split}.json'
    if result.get('ground_truth_sha256') != sha256_file(ground_truth):
        raise ValueError(f'Ground-truth checksum mismatch: {path}')
    ref_scores = read_json(reference)[split]
    ref = result.get('reference', {})
    if Path(ref.get('path', '')).resolve() != reference or ref.get('sha256') != sha256_file(reference) or ref.get('split') != split:
        raise ValueError(f'Reference provenance mismatch: {path}')
    for name in ('macro_f1', 'mAP'):
        if ref.get(name) != ref_scores[name]:
            raise ValueError(f'Reference score mismatch: {name}')
    for name in ('accuracy', 'macro_f1', 'mAP', 'AP50'):
        score = finite_score(result['metrics'][name])
        if not 0 <= score <= 1:
            raise ValueError(f'Expected fraction-valued metric: {name}')
    score = finite_score(result['local_delta_m'])
    if not math.isclose(score, local_delta_m(result['metrics'], ref_scores), rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f'Local delta_m calculation mismatch: {path}')
    return result


def evaluation_command(python, run, data, reference, split, output):
    return [str(python), 'tools/evaluate_lr_checkpoint.py', '--run', str(run), '--weights', 'model.pt',
            '--data', str(data), '--split', split, '--reference', str(reference),
            '--batch-size', '4', '--workers', '2', '--out', str(output)]


def configure_resources():
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1', NCCL_P2P_DISABLE='1', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                      TORCHINDUCTOR_COMPILE_THREADS='1', TOKENIZERS_PARALLELISM='false',
                      PYTHONUNBUFFERED='1', HF_HOME='/mnt/hdd/junkaichen/hf-cache', HF_HUB_OFFLINE='1')
    for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'):
        os.environ.pop(name, None)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--dependency-suite', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    suite, dependency, data, reference = [value.resolve() for value in
                                        (args.suite, args.dependency_suite, args.data, args.reference)]
    if suite == dependency:
        raise SystemExit('Follow-up must use a separate suite directory')
    lock = (suite / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite / 'status.json').exists():
        raise SystemExit('Suite already started; refusing to overwrite it')
    configure_resources()
    state = {'pid': os.getpid(), 'started_at': stamp(), 'suite': str(suite),
             'dependency_suite': str(dependency), 'gpus': [0, 1],
             'cpu_affinity': sorted(os.sched_getaffinity(0)), 'global_batch': 16,
             'maximum_additional_training_runs': 2, 'completed_training_runs': [],
             'official_submissions': 0, 'automatic_full_data_retrain': False}
    commands = {}
    dependency_lock = None

    def update(stage=None, **extra):
        if stage is not None:
            state['stage'] = stage
        state.update(updated_at=stamp(), **extra)
        write_json(suite / 'status.json', state)
        if stage is not None:
            print(stamp(), stage, json.dumps(extra), flush=True)

    def execute(name, command, log_path, single_gpu=False):
        commands[name] = command
        write_json(suite / 'commands.json', commands)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0') if single_gpu else None
        print(stamp(), 'COMMAND', shlex.join(command), flush=True)
        with Path(log_path).open('a') as output:
            with subprocess.Popen(command, cwd=suite / 'code', env=environment,
                                  stdout=output, stderr=subprocess.STDOUT) as child:
                update(child_pid=child.pid)
                while True:
                    try:
                        returncode = child.wait(timeout=POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        update()
        state.pop('child_pid', None)
        update()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)

    try:
        update('verifying_inputs')
        if ROOT.resolve() != (suite / 'code').resolve():
            raise RuntimeError('Run this driver from the suite code snapshot')
        verification = verify_snapshot(suite, dependency, data, reference)
        write_json(suite / 'input_verification.json', {'time': stamp(), **verification})
        update('waiting_for_dependency')
        deadline = time.monotonic() + WAIT_SECONDS
        dependency_lock = (dependency / 'driver.lock').open('r')
        while True:
            previous = read_json(dependency / 'status.json')
            acquired = False
            if previous.get('stage') == 'complete':
                try:
                    fcntl.flock(dependency_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    pass
            ready = dependency_ready(previous, locked=not acquired,
                                     alive=process_is_dependency(previous.get('pid'), dependency))
            update(dependency_stage=previous.get('stage'), dependency_updated_at=previous.get('updated_at'))
            if ready:
                break  # keep the acquired dependency lock for this entire follow-up
            if time.monotonic() >= deadline:
                raise TimeoutError('Dependency did not finish and release its lock within 48 hours')
            time.sleep(POLL_SECONDS)
        update('validating_dependency_results')
        # Recheck the frozen implementation and manifests after the potentially long wait.
        verify_snapshot(suite, dependency, data, reference)
        trials = {
            'lr1x': {'run': dependency / 'lr1x', 'head_lr': 1e-4, 'backbone_lr': 3e-4, 'epochs': 130},
            'lr05x': {'run': dependency / 'lr05x', 'head_lr': 5e-5, 'backbone_lr': 1.5e-4, 'epochs': 130},
            'hybrid130': {'run': suite / 'hybrid130', 'head_lr': 1e-4, 'backbone_lr': 2e-4, 'epochs': 130},
        }
        results = {}
        for name in ('lr1x', 'lr05x'):
            trial = trials[name]
            results[name] = validate_result(trial['run'], trial['run'] / 'metrics_submission_post_val.json',
                                            data=data, reference=reference, **{k: trial[k] for k in ('epochs', 'head_lr', 'backbone_lr')})

        def run_trial(name):
            trial = trials[name]
            run = trial['run']
            run.mkdir()  # refuse any existing trial rather than silently overwriting it
            update('training', active_trial=name)
            command = build_training_command(sys.executable, data, suite, name,
                                             trial['head_lr'], trial['backbone_lr'], trial['epochs'])
            execute(name, command, run / 'stdout.log')
            state['completed_training_runs'].append(name)
            update('evaluating_validation', active_trial=name)
            output = run / 'metrics_submission_post_val.json'
            execute(name + '_val', evaluation_command(sys.executable, run, data, reference, 'val', output),
                    run / 'evaluation.log', single_gpu=True)
            results[name] = validate_result(run, output, data=data, reference=reference,
                                            **{k: trial[k] for k in ('epochs', 'head_lr', 'backbone_lr')})
            update('trial_complete', active_trial=name)

        run_trial('hybrid130')
        lr_scores = {name: results[name]['local_delta_m'] for name in LR_PREFERENCE}
        chosen_lr = choose_lr_candidate(lr_scores)
        selection = {'time': stamp(), 'selection_split': 'val', 'endpoint': 'epoch130 submission-post local_delta_m',
                     'scores': lr_scores, 'selected_lr': chosen_lr, 'indifference_margin': MARGIN,
                     'preference': list(LR_PREFERENCE),
                     'policy': 'Scores within 1.0 local delta_m of the maximum share a tier; prefer the declared order.',
                     'caution': 'Operational margin, not statistical significance; all candidates use seed0.',
                     'planned_duration_epochs': 160, 'test_used_for_selection': False}
        write_json(suite / 'lr_selection.json', selection)
        chosen = trials[chosen_lr]
        trials['schedule160'] = {'run': suite / 'schedule160', 'head_lr': chosen['head_lr'],
                                 'backbone_lr': chosen['backbone_lr'], 'epochs': 160}
        run_trial('schedule160')
        final_name = choose_duration(chosen_lr, results[chosen_lr]['local_delta_m'], results['schedule160']['local_delta_m'])
        improvement = results['schedule160']['local_delta_m'] - results[chosen_lr]['local_delta_m']
        candidate = {'time': stamp(), 'candidate': final_name, 'run': str(trials[final_name]['run']),
                     'selection_split': 'val', 'short_candidate': chosen_lr,
                     'duration_gain_delta_m': improvement, 'longer_duration_requires_gain_strictly_above': MARGIN,
                     'local_val_delta_m': results[final_name]['local_delta_m'],
                     'test_used_for_selection': False, 'status': 'exploratory_single_seed'}
        write_json(suite / 'candidate.json', candidate)  # fixed BEFORE either confirmation test is evaluated
        test_results = {}
        for name in dict.fromkeys((final_name, 'lr1x')):
            trial = trials[name]
            update('confirming_local_test', active_trial=name)
            # Keep new evaluations in this suite; do not write into the existing pair's runs.
            output = suite / f'confirmation_test_{name}.json'
            execute(name + '_test', evaluation_command(sys.executable, trial['run'], data, reference, 'test', output),
                    suite / f'confirmation_test_{name}.log', single_gpu=True)
            test_results[name] = validate_result(trial['run'], output, data=data, reference=reference, split='test',
                                                 **{k: trial[k] for k in ('epochs', 'head_lr', 'backbone_lr')})
        val_gain = results[final_name]['local_delta_m'] - results['lr1x']['local_delta_m']
        test_gain = test_results[final_name]['local_delta_m'] - test_results['lr1x']['local_delta_m']
        report = {'time': stamp(), 'lr_selection': selection, 'candidate_fixed_before_test': candidate,
                  'validation_results': results, 'confirmation_test_results': test_results,
                  'candidate_minus_baseline_val_delta_m': val_gain,
                  'candidate_minus_baseline_test_delta_m': test_gain,
                  'duration_gain_delta_m': improvement,
                  'test_direction_agrees_with_val': val_gain * test_gain >= 0,
                  'test_note': 'This local test was used historically; it is not a pristine test set. '
                               'Its result did not alter candidate selection.',
                  'status': 'exploratory_single_seed', 'additional_training_runs': 2,
                  'paired_seed_confirmation_recommended': final_name != 'lr1x',
                  'next_step': ('If pursuing the candidate, train baseline and candidate with a second matched seed '
                                'before claiming a stable gain or choosing a full-data retrain.'
                                if final_name != 'lr1x' else 'Retain the baseline; no new candidate cleared the '
                                'predeclared operational margin. No automatic additional training.'),
                  'official_submissions': 0, 'automatic_full_data_retrain': False}
        write_json(suite / 'comparison.json', report)
        update('complete', active_trial=None, candidate=final_name,
               candidate_minus_baseline_val_delta_m=val_gain,
               candidate_minus_baseline_test_delta_m=test_gain)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if dependency_lock is not None:
            dependency_lock.close()


if __name__ == '__main__':
    main()
