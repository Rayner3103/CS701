"""Local BP7 classification-only context; matched e150 comparison, final e160 evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from run_bp7_lr125_trial import process_identity
from run_bp7_schedule_search import (build_training_command, configure_resources, evaluation_command,
                                    read_json, sha256_file, stamp, verify_snapshot, write_json)

TOTAL, TRAINABLE = 125126304, 39916704


def training_command(python, data, suite, name='context160'):
    return build_training_command(python, data, suite, name, 1e-4, 3e-4, 160) + [
        '--roi-context', '3', '--roi-context-mode', 'classification']


def validate_training_metrics(metrics, expected_args):
    if metrics['args'] != expected_args:
        raise ValueError('Training arguments differ from declared command')
    if metrics['parameters'] != TOTAL or metrics['trainable_parameters'] != TRAINABLE:
        raise ValueError('Unexpected context model parameter counts')
    if metrics['parameters'] + metrics.get('buffer_values', 0) > 130000000 or metrics['trainable_parameters'] > 40000000:
        raise ValueError('Parameter budget exceeded')


def validate_evaluation(result, *, expected_args, weights_sha256, post):
    if result['training_args'] != expected_args:
        raise ValueError('Evaluation uses another training recipe')
    if result['weights_sha256'] != weights_sha256:
        raise ValueError('Evaluation checkpoint hash mismatch')
    if result['post'] != post or result['split'] != 'val' or result['n_images'] != 940 or result['training_splits'] != ['train']:
        raise ValueError('Evaluation postprocessing or holdout differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    suite = parser.parse_args().suite.resolve()
    manifest = read_json(suite/'launch_manifest.json')
    data, reference = Path(manifest['data']), Path(manifest['reference'])
    if Path(__file__).resolve().parents[1] != suite/'code':
        raise RuntimeError('Use frozen snapshot')
    lock = (suite/'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite/'status.json').exists():
        raise RuntimeError('Refusing to overwrite existing trial')
    configure_resources()
    state = dict(pid=os.getpid(), started_at=stamp(), suite=str(suite), gpus=[0, 1],
                 cpu_affinity=sorted(os.sched_getaffinity(0)), global_batch=16, epochs=160,
                 training_images=7512, maximum_new_training_runs=1, official_submissions=0,
                 automatic_full_data_retrain=False)
    commands, held_locks = {}, []

    def update(stage=None, **extra):
        if stage:
            state['stage'] = stage
            print(stamp(), stage, extra, flush=True)
        state.update(updated_at=stamp(), **extra)
        write_json(suite/'status.json', state)

    def execute(name, command, log, single_gpu=False):
        commands[name] = command
        write_json(suite/'commands.json', commands)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0') if single_gpu else None
        with log.open('a') as stream:
            with subprocess.Popen(command, cwd=suite/'code', env=env, stdout=stream, stderr=subprocess.STDOUT) as child:
                update(child_pid=child.pid)
                while True:
                    try:
                        rc = child.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        update()
        state.pop('child_pid', None)
        if rc:
            raise subprocess.CalledProcessError(rc, command)

    def parsed(command):
        import train
        return json.loads(json.dumps(vars(train.parse_args(command[command.index('train.py')+1:])), default=str))

    try:
        update('verifying_inputs')
        checks = verify_snapshot(suite, Path(manifest['dependency_suite']), data, reference)
        write_json(suite/'input_verification.json', checks)
        for path in map(Path, manifest['stopped_suites']):
            old = read_json(path/'status.json')
            if old['stage'] != 'stopped_by_user':
                raise RuntimeError('Previous experiment was not explicitly stopped')
            fd = (path/'driver.lock').open('a')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held_locks.append(fd)
            if process_identity(old['pid']) is not None:
                raise RuntimeError('Previous controller still active')
            for proc in Path('/proc').iterdir():
                if proc.name.isdigit():
                    identity = process_identity(int(proc.name))
                    if identity and 'train.py' in identity['command'] and str(path) in identity['command']:
                        raise RuntimeError('Previous training still active')
        baseline = Path(manifest['matched_baseline_run'])
        if sha256_file(baseline/'ckpt_e150.pt') != manifest['matched_baseline_sha256']:
            raise RuntimeError('Matched baseline checkpoint changed')
        update('smoke_training')
        smoke_root = suite/'validation'
        smoke_root.mkdir()
        smoke = training_command(sys.executable, data, smoke_root, 'smoke')
        for flag, value in {'--epochs':'1', '--schedule-epochs':'1', '--eval-splits':'',
                            '--save-every':'0', '--workers':'0'}.items():
            smoke[smoke.index(flag)+1] = value
        smoke += ['--limit', '32']
        execute('smoke', smoke, smoke_root/'smoke.log')
        validate_training_metrics(read_json(smoke_root/'smoke/metrics.json'), parsed(smoke))
        write_json(smoke_root/'checks.json', dict(parameter_budget_passed=True,
                   total=TOTAL, trainable=TRAINABLE, training_steps=2, two_gpu_smoke_passed=True))
        run = suite/'context160'
        run.mkdir()
        command = training_command(sys.executable, data, suite)
        update('training', active_trial='context160')
        execute('context160', command, run/'stdout.log')
        validate_training_metrics(read_json(run/'metrics.json'), parsed(command))
        from evaluate_lr_checkpoint import local_delta_m, postprocessing_arguments
        from predict import saved_args
        results = {}
        for label, source, weights in [('baseline_e150', baseline, 'ckpt_e150.pt'),
                                       ('context_e150', run, 'ckpt_e150.pt'),
                                       ('context_e160', run, 'model.pt')]:
            update('evaluating_validation', evaluation=label)
            destination = suite/f'{label}_post_val.json'
            evaluation = evaluation_command(sys.executable, source, data, reference, 'val', destination)
            evaluation[evaluation.index('--weights')+1] = weights
            execute(label, evaluation, suite/f'{label}_evaluation.log', single_gpu=True)
            result = read_json(destination)
            validate_evaluation(result, expected_args=saved_args(source),
                                weights_sha256=sha256_file(source/weights), post=vars(postprocessing_arguments()))
            if result['ground_truth_sha256'] != sha256_file(data/'detection/instances_val.json'):
                raise RuntimeError('Ground truth hash differs')
            if result['reference']['sha256'] != sha256_file(reference):
                raise RuntimeError('Reference hash differs')
            score = local_delta_m(result['metrics'], read_json(reference)['val'])
            if abs(score-result['local_delta_m']) > 1e-8:
                raise RuntimeError('Local delta_m differs')
            results[label] = result
        gain = results['context_e150']['local_delta_m']-results['baseline_e150']['local_delta_m']
        write_json(suite/'comparison.json', dict(time=stamp(), results=results,
                   primary_comparison='e150 vs e150, both 160-epoch cosine, train7512, seed0',
                   matched_context_minus_baseline_local_delta_m=gain,
                   note='Single seed; context changes initialization RNG consumption. '
                        'Old e160 baseline stopped by user; new e160 has no matched completed baseline.',
                   automatic_further_training=False, official_submissions=0))
        update('complete', active_trial=None, matched_context_minus_baseline_local_delta_m=gain)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for fd in held_locks:
            fd.close()


if __name__ == '__main__':
    main()
