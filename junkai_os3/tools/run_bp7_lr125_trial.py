"""Adopt the running hybrid trial, then run one fresh 1.25x LR trial.

This explicitly supersedes the old 160-epoch queue without interrupting training.
Only local validation is evaluated; no upload, test selection, or further training.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_bp7_schedule_search import (
    build_training_command, configure_resources, evaluation_command, read_json,
    stamp, validate_result, verify_snapshot, write_json,
)


def process_identity(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'command': (root / 'cmdline').read_bytes().decode().split('\0')[:-1]}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--previous-suite', type=Path, required=True)
    args = parser.parse_args()
    suite, previous = args.suite.resolve(), args.previous_suite.resolve()
    manifest = read_json(suite / 'launch_manifest.json')
    dependency = Path(manifest['dependency_suite'])
    data, reference = Path(manifest['data']), Path(manifest['reference'])
    if Path(__file__).resolve().parents[1] != suite / 'code':
        raise RuntimeError('Use the frozen source snapshot')
    lock = (suite / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite / 'status.json').exists():
        raise RuntimeError('Refusing to restart or overwrite an existing suite')
    configure_resources()
    state = {'pid': os.getpid(), 'started_at': stamp(), 'suite': str(suite),
             'previous_suite': str(previous), 'maximum_new_training_runs': 1,
             'deferred_trial': 'schedule160', 'gpus': [0, 1],
             'cpu_affinity': sorted(os.sched_getaffinity(0)), 'global_batch': 16,
             'official_submissions': 0, 'automatic_full_data_retrain': False}
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

    previous_lock = None
    try:
        update('verifying_inputs')
        checks = verify_snapshot(suite, dependency, data, reference)
        baseline = validate_result(dependency / 'lr1x', dependency / 'lr1x/metrics_submission_post_val.json',
                                   data=data, reference=reference, epochs=130, head_lr=1e-4, backbone_lr=3e-4)
        old = read_json(previous / 'status.json')
        if old['stage'] != 'training' or old['active_trial'] != 'hybrid130' or (previous / 'schedule160').exists():
            raise RuntimeError('Old queue changed; inspect before takeover')
        parent = process_identity(old['pid'])
        expected_driver = str(previous / 'code/tools/run_bp7_schedule_search.py')
        if not parent or expected_driver not in parent['command']:
            raise RuntimeError('Old driver identity mismatch')
        child = process_identity(old['child_pid'])
        if not child or 'train.py' not in child['command'] or str(previous) not in child['command']:
            raise RuntimeError('Training child identity mismatch')
        # Capture rank identities too: never overlap GPU jobs if torchrun dies early.
        ranks = []
        for proc in Path('/proc').iterdir():
            if proc.name.isdigit():
                identity = process_identity(int(proc.name))
                if identity and 'train.py' in identity['command'] and str(previous) in identity['command']:
                    ranks.append(identity)
        if child not in ranks or len(ranks) < 3:
            raise RuntimeError('Could not identify torchrun and both training ranks')
        write_json(suite / 'handoff.json', {'time': stamp(), 'old_status': old,
                   'old_driver': parent, 'training_processes': ranks, 'verification': checks,
                   'reason': 'User selected 1.25x LR at 130 epochs before any 160-epoch trial.'})
        if process_identity(parent['pid']) != parent:
            raise RuntimeError('Old driver identity changed before handoff')
        # SIGTERM only the controller PID; never signal its process group or ranks.
        os.kill(parent['pid'], signal.SIGTERM)
        previous_lock = (previous / 'driver.lock').open('a')
        deadline = time.monotonic() + 30
        while True:
            try:
                fcntl.flock(previous_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Previous controller lock did not release')
                time.sleep(1)
        old.update(stage='superseded', updated_at=stamp(), superseded_by=str(suite),
                   deferred_trial='schedule160', training_continues=True)
        write_json(previous / 'status.json', old)
        update('waiting_for_hybrid130', adopted_child_pid=child['pid'])
        deadline = time.monotonic() + 48 * 3600
        while any(process_identity(item['pid']) == item for item in ranks):
            if time.monotonic() > deadline:
                raise TimeoutError('Adopted training exceeded 48 hours; no overlapping job launched')
            time.sleep(30)
            update()
        hybrid = previous / 'hybrid130'
        if not (hybrid / 'metrics.json').exists() or not (hybrid / 'model.pt').exists():
            raise RuntimeError('Adopted training exited without final outputs')
        import torch
        checkpoint = torch.load(hybrid / 'training_state.pt', map_location='cpu', weights_only=False)
        if checkpoint['epoch'] != 130 or checkpoint['global_step'] != 130 * 470:
            raise RuntimeError('Adopted training did not finish all 130 epochs')
        del checkpoint
        verify_snapshot(suite, dependency, data, reference)
        update('evaluating_hybrid130')
        hybrid_output = hybrid / 'metrics_submission_post_val.json'
        execute('hybrid130_val', evaluation_command(sys.executable, hybrid, data, reference, 'val', hybrid_output),
                suite / 'hybrid130_evaluation.log', single_gpu=True)
        hybrid_result = validate_result(hybrid, hybrid_output, data=data, reference=reference,
                                       epochs=130, head_lr=1e-4, backbone_lr=2e-4)
        old.update(updated_at=stamp(), training_continues=False, completed_training_runs=['hybrid130'])
        write_json(previous / 'status.json', old)
        run = suite / 'lr125x'
        run.mkdir()
        update('training', active_trial='lr125x')
        execute('lr125x', build_training_command(sys.executable, data, suite, 'lr125x', 1.25e-4, 3.75e-4, 130),
                run / 'stdout.log')
        update('evaluating_validation', active_trial='lr125x')
        output = run / 'metrics_submission_post_val.json'
        execute('lr125x_val', evaluation_command(sys.executable, run, data, reference, 'val', output),
                run / 'evaluation.log', single_gpu=True)
        result = validate_result(run, output, data=data, reference=reference,
                                 epochs=130, head_lr=1.25e-4, backbone_lr=3.75e-4)
        gain = result['local_delta_m'] - baseline['local_delta_m']
        write_json(suite / 'comparison.json', {'time': stamp(), 'baseline': baseline,
                   'hybrid130': hybrid_result, 'lr125x': result,
                   'lr125x_minus_baseline_local_delta_m': gain,
                   'metric_differences_pp': {k: 100 * (result['metrics'][k] - baseline['metrics'][k])
                                             for k in ('mAP', 'macro_f1', 'accuracy', 'AP50')},
                   'status': 'exploratory_single_seed', 'selection_split': 'val',
                   'test_used': False, 'schedule160': 'deferred', 'automatic_further_training': False})
        update('complete', active_trial=None, lr125x_minus_baseline_local_delta_m=gain)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if previous_lock is not None:
            previous_lock.close()


if __name__ == '__main__':
    main()
