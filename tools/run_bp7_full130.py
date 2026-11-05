"""Full-data two-GPU BP7: SC2 capacity plus SpatialPrior(64), 130-epoch cosine.

Run only from a fixed snapshot. Official validation images are for inference.
The schedule differs from SC2's epoch 130 of a 150-epoch schedule.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo


def stamp():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def training_arguments(data):
    return [
        '--data', str(data), '--train-split', 'train+val+test', '--eval-splits', '',
        '--epochs', '130', '--save-every', '10', '--seed', '0',
        '--batch-size', '16', '--workers', '2',
        '--backbone', 'dinov3', '--init', 'pretrained', '--adapt', 'lora',
        '--lora-targets', 'attn.qkv,attn.proj,mlp.fc1,mlp.fc2',
        '--lora-rank', '80', '--lora-alpha', '160', '--cascade-rep', '1536',
        '--train-norms', '--backbone-lr', '3e-4', '--lr', '1e-4',
        '--weight-decay', '0.05', '--compile', '--cascade', '--aligned-roi',
        '--rpn-nwd', '8', '--anchor-offset', '0.5',
        '--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181',
        '--cascade-size-iou', '--flip-bias', '-0.3', '--grad-ckpt',
        '--spatial-prior', '64',
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    code = run / 'code'
    lock = (run / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((run / name).exists() for name in ['status.json', 'log.txt', 'model.pt']):
        raise SystemExit('Run already started; refusing to overwrite or restart it.')
    os.environ.update(
        CUDA_VISIBLE_DEVICES='0,1', NCCL_P2P_DISABLE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2', TORCHINDUCTOR_COMPILE_THREADS='1',
        TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
        HF_HOME='/mnt/hdd/junkaichen/hf-cache', HF_HUB_OFFLINE='1',
    )
    for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
                'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK',
                'ROLE_WORLD_SIZE']:
        os.environ.pop(key, None)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    state = {
        'pid': os.getpid(), 'started_at': stamp(), 'run': str(run),
        'gpus': [0, 1], 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'epochs': 130, 'global_batch': 16, 'data_workers_per_rank': 2,
    }

    def update(stage, **extra):
        state.update(stage=stage, updated_at=stamp(), **extra)
        write_json(run / 'status.json', state)
        print(stamp(), stage, json.dumps(extra), flush=True)

    def execute(command, log_path, env=None):
        print(stamp(), 'COMMAND', shlex.join(command), flush=True)
        with log_path.open('a') as output:
            with subprocess.Popen(command, cwd=code, env=env, stdout=output,
                                  stderr=subprocess.STDOUT) as child:
                state['child_pid'] = child.pid
                write_json(run / 'status.json', state)
                returncode = child.wait()
        state.pop('child_pid', None)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)

    ddp = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node', '2', 'train.py']
    base = training_arguments(args.data)
    validation = run / 'validation'
    training = ddp + base + ['--out', str(run.parent), '--name', run.name]
    try:
        update('verifying_snapshot')
        for line in (run / 'source.sha256').read_text().splitlines():
            expected, name = line.split('  ', 1)
            actual = hashlib.sha256((code / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Source checksum mismatch: {name}')
        validation.mkdir(exist_ok=True)
        smoke_root = validation / 'smoke'
        smoke = smoke_root / 'ddp_smoke'
        if smoke.exists():
            raise RuntimeError('Existing smoke run; refusing to overwrite.')
        smoke_command = ddp + base + [
            '--out', str(smoke_root), '--name', smoke.name, '--epochs', '1',
            '--limit', '48', '--workers', '0', '--save-every', '0',
        ]
        write_json(run / 'commands.json', {'smoke': smoke_command, 'training': training})
        update('smoke_training')
        execute(smoke_command, validation / 'smoke_train.log')
        metrics = json.loads((smoke / 'metrics.json').read_text())
        assert metrics['parameters'] == 125076960, metrics['parameters']
        assert metrics['trainable_parameters'] == 39867360, metrics['trainable_parameters']
        assert metrics['args']['spatial_prior'] == 64
        assert not metrics['args']['pretrained_prior']
        assert (smoke / 'model.pt').stat().st_size > 0
        single_gpu = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
        images = json.loads((args.images / 'images.json').read_text())
        images['images'] = images['images'][:8]
        image_list = validation / 'smoke_images.json'
        write_json(image_list, images)
        smoke_zip = validation / 'smoke_submission.zip'
        update('smoke_checkpoint_prediction')
        execute([sys.executable, 'predict.py', '--run', str(smoke),
                 '--images', str(args.images), '--images-json', str(image_list),
                 '--batch-size', '4', '--workers', '0', '--shift', '-0.2',
                 '--out', str(smoke_zip)], validation / 'smoke_predict.log', single_gpu)
        execute([sys.executable, 'tools/check_submission.py', str(smoke_zip), str(image_list)],
                validation / 'smoke_predict.log', single_gpu)
        write_json(validation / 'smoke_passed.json', {
            'time': stamp(), 'parameters': metrics['parameters'],
            'trainable_parameters': metrics['trainable_parameters'],
            'rank0_peak_gpu_memory_gb': metrics['peak_gpu_memory_gb'],
            'ddp_steps': 3, 'prediction_images': 8,
        })
        update('training')
        execute(training, run / 'stdout.log')
        final = json.loads((run / 'metrics.json').read_text())
        assert final['args']['epochs'] == 130
        assert (run / 'model.pt').stat().st_size > 0
        update('predicting_phase1')
        submissions = run / 'submission'
        submissions.mkdir()
        destination = submissions / f'{run.name}_e130_val_shift-0.2.zip'
        execute([sys.executable, 'predict.py', '--run', str(run),
                 '--images', str(args.images), '--batch-size', '8', '--workers', '2',
                 '--shift', '-0.2', '--out', str(destination),
                 '--raw-out', str(submissions / 'val_raw.pt')], run / 'prediction.log', single_gpu)
        execute([sys.executable, 'tools/check_submission.py', str(destination),
                 str(args.images / 'images.json')], run / 'prediction.log', single_gpu)
        checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
        (submissions / 'SHA256SUMS').write_text(f'{checksum}  {destination.name}\n')
        update('complete', submission=str(destination), uploaded=False)
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
