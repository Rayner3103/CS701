"""Run the next bridge/playground screens after the existing queue exits.

The code directory must be a fixed snapshot. Only GPU0 is exposed; all jobs run
serially. A failed prerequisite, diagnostic or smoke test stops the queue.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo


EXPERIMENTS = [
    ('BP7_spatial64', ['--spatial-prior', '64']),
    ('BP8_ctx3_cls_only', ['--roi-context', '3', '--roi-context-mode', 'classification']),
    ('BP9_ctx3_clsw3', ['--roi-context', '3', '--cls-weights', '3:3,7:3']),
]


def stamp():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')


def process_matches(pid, start_ticks):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == start_ticks
    except FileNotFoundError:
        return False


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def base_arguments(data, runs):
    return [
        '--data', str(data), '--out', str(runs), '--epochs', '36', '--dump-raw',
        '--workers', '4', '--seed', '0', '--batch-size', '16',
        '--backbone', 'dinov3', '--init', 'pretrained', '--adapt', 'lora',
        '--lora-targets', 'attn.qkv,attn.proj,mlp.fc1,mlp.fc2',
        '--lora-rank', '32', '--lora-alpha', '64', '--train-norms',
        '--backbone-lr', '3e-4', '--compile', '--cascade', '--aligned-roi',
        '--rpn-nwd', '8', '--anchor-offset', '0.5',
        '--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181',
        '--cascade-size-iou', '--flip-bias', '-0.3', '--grad-ckpt',
    ]


def comparison(runs):
    rows = []
    names = ['AA_a4a2', 'BP1_ctx3', 'BP2_clsw3', 'BP4_partial',
             'BP5_partial_ctx3', 'BP3_os2'] + [name for name, _ in EXPERIMENTS]
    refs = {'val': (91.50, 22.41), 'test': (91.61, 23.31)}
    for name in names:
        path = runs / name / 'metrics_pp.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        val, test = result['val'], result['test']
        delta = []
        for split, metrics in [('val', val), ('test', test)]:
            f_ref, m_ref = refs[split]
            delta.append(50 * ((100 * metrics['macro_f1'] - f_ref) / f_ref
                               + (100 * metrics['mAP'] - m_ref) / m_ref))
        row = {
            'run': name, 'post': result['post'], 'local_delta_m': sum(delta) / 2,
            'val_macro_f1': 100 * val['macro_f1'], 'test_macro_f1': 100 * test['macro_f1'],
            'val_mAP': 100 * val['mAP'], 'test_mAP': 100 * test['mAP'],
            'per_class_AP_mean': {c: 50 * (val['per_class_AP'][c] + test['per_class_AP'][c])
                                  for c in val['per_class_AP']},
        }
        fusion = runs / name / 'ppcls.json'
        if fusion.exists():
            row['postprocessing_variants'] = json.loads(fusion.read_text())
        rows.append(row)
    return {'updated_at': stamp(), 'reference_percent': refs,
            'note': 'Local holdouts, not leaderboard scores. Compare only matching post configurations.',
            'runs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--predecessor-pid', type=int, required=True)
    parser.add_argument('--predecessor-start-ticks', type=int, required=True)
    parser.add_argument('--predecessor-log', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    base = base_arguments(args.data, args.runs)
    if args.plan_only:
        print(json.dumps({'diagnostic': 'AA training split, 16 images/class, no weight updates',
                          'experiments': [{'name': n, 'argv': base + ['--name', n] + extra}
                                          for n, extra in EXPERIMENTS]}, indent=2))
        return

    args.queue_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.queue_dir / 'queue.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This follow-up queue is already running.')
    if (args.queue_dir / 'status.json').exists():
        raise SystemExit('Existing queue state found; inspect it before restarting to avoid overwriting results.')
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
                      TORCHINDUCTOR_COMPILE_THREADS='2', TOKENIZERS_PARALLELISM='false',
                      WORLD_SIZE='1', RANK='0', LOCAL_RANK='0',
                      HF_HOME='/mnt/hdd/junkaichen/hf-cache', HF_HUB_OFFLINE='1')
    # Bound compilation/data-loading CPU use as well as individual library thread pools.
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[:4])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    state = {'pid': os.getpid(), 'started_at': stamp(), 'gpu': 0,
             'cpu_affinity': sorted(os.sched_getaffinity(0)),
             'predecessor_pid': args.predecessor_pid, 'experiments': [n for n, _ in EXPERIMENTS]}

    def update(stage, **extra):
        state.update(stage=stage, updated_at=stamp(), **extra)
        write_json(args.queue_dir / 'status.json', state)
        print(stamp(), stage, json.dumps(extra), flush=True)

    def execute(command, cwd, log_path):
        print(stamp(), 'COMMAND', shlex.join(command), flush=True)
        with log_path.open('a') as output:
            subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True)

    try:
        update('waiting_predecessor')
        while process_matches(args.predecessor_pid, args.predecessor_start_ticks):
            time.sleep(30)
        if 'QUEUE2 DONE' not in args.predecessor_log.read_text():
            raise RuntimeError('Previous queue exited without its completion marker; inspect it before continuing.')
        for name in ['BP5_partial_ctx3', 'BP3_os2']:
            for filename in ['metrics_pp.json', 'ppcls.json']:
                if not (args.runs / name / filename).is_file():
                    raise RuntimeError(f'Prerequisite {name}/{filename} is missing; refusing to overlap/retry it.')
        for name, _ in EXPERIMENTS:
            if (args.runs / name).exists():
                raise RuntimeError(f'{name} already exists; refusing to overwrite an experiment.')
        # Verify the copied source before allowing any GPU job to begin.
        hashes = args.queue_dir / 'source.sha256'
        execute(['sha256sum', '-c', str(hashes)], args.code, args.queue_dir / 'source_check.log')
        update('diagnostic')
        execute([sys.executable, 'tools/diagnose_roi_samples.py', '--run', str(args.runs / 'AA_a4a2'),
                 '--data', str(args.data), '--out', str(args.queue_dir / 'roi_diagnostic.json'),
                 '--per-class', '16', '--batch-size', '2', '--device', 'cuda:0', '--seed', '0'],
                args.code, args.queue_dir / 'diagnostic.log')
        for name, extra in EXPERIMENTS:
            update('smoke_test', experiment=name)
            smoke_root = args.queue_dir / 'smoke'
            smoke_base = [argument for argument in base if argument != '--compile']
            smoke_args = smoke_base + ['--out', str(smoke_root), '--name', name,
                                       '--epochs', '1', '--limit', '16', '--batch-size', '2', '--workers', '0'] + extra
            execute([sys.executable, 'train.py'] + smoke_args, args.code, args.queue_dir / f'{name}_smoke.log')
            execute([sys.executable, 'predict.py', '--run', str(smoke_root / name),
                     '--data', str(args.data), '--split', 'val', '--batch-size', '8', '--workers', '0'],
                    args.code, args.queue_dir / f'{name}_smoke.log')
            run = args.runs / name
            run.mkdir()
            shutil.copytree(args.code, run / 'code', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
            command = [sys.executable, 'train.py'] + base + ['--name', name] + extra
            write_json(run / 'launch.json', {'time': stamp(), 'argv': command, 'source': str(args.code)})
            update('training', experiment=name)
            execute(command, run / 'code', run / 'stdout.log')
            update('evaluation', experiment=name)
            execute([sys.executable, 'pp_eval.py', '--run', str(run)], run / 'code', run / 'stdout.log')
            execute([sys.executable, 'tools/evaluate_bp_fusion.py', '--run', str(run)],
                    run / 'code', run / 'stdout.log')
            write_json(args.queue_dir / 'comparison.json', comparison(args.runs))
            update('experiment_complete', experiment=name)
        update('complete', experiment=None)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
