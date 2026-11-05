"""One fresh BP7 trial: original peak learning rates, 160-epoch cosine schedule."""
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
                 official_submissions=0, automatic_full_data_retrain=False)
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
            expected_driver = str(previous / 'code/tools/run_bp7_lr125_trial.py')
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
        run = suite / 'schedule160'
        run.mkdir()
        update('training', active_trial='schedule160')
        execute('schedule160', build_training_command(sys.executable, data, suite, 'schedule160',
                                                      1e-4, 3e-4, 160), run / 'stdout.log')
        update('evaluating_validation')
        output = run / 'metrics_submission_post_val.json'
        execute('schedule160_val', evaluation_command(sys.executable, run, data, reference, 'val', output),
                run / 'evaluation.log', single_gpu=True)
        result = validate_result(run, output, data=data, reference=reference,
                                 epochs=160, head_lr=1e-4, backbone_lr=3e-4)
        gain = result['local_delta_m'] - baseline['local_delta_m']
        write_json(suite / 'comparison.json', dict(time=stamp(), baseline=baseline, schedule160=result,
                   schedule160_minus_baseline_local_delta_m=gain,
                   metric_differences_pp={k: 100 * (result['metrics'][k] - baseline['metrics'][k])
                                          for k in ('mAP', 'macro_f1', 'accuracy', 'AP50')},
                   clears_operational_margin=gain > 1.0,
                   caution='Single seed; 1.0 is an operational margin, not statistical significance. '
                           'Longer duration also stretches the entire cosine LR trajectory.',
                   test_used=False, automatic_further_training=False))
        update('complete', active_trial=None, schedule160_minus_baseline_local_delta_m=gain)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if dependency_lock is not None:
            dependency_lock.close()


if __name__ == '__main__':
    main()
