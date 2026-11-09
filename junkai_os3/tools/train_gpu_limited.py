"""Run frozen train/predict entry points with a per-process CUDA allocator limit."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-fraction', type=float, required=True)
    parser.add_argument('script', choices=('train.py', 'predict.py'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not 0 < args.memory_fraction <= 0.20:
        parser.error('The concurrent run permits at most 0.20 of device memory per rank')
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch

    # LOCAL_RANK is indexed within CUDA_VISIBLE_DEVICES. A single-GPU predictor
    # has no distributed environment and therefore uses its visible device 0.
    device = int(os.environ.get('LOCAL_RANK', '0'))
    if not 0 <= device < torch.cuda.device_count():
        raise ValueError(f'Invalid local CUDA rank: {device}')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.argv = [str(root / args.script), *args.arguments]
    print('GPU_RESOURCE_LIMIT ' + json.dumps(dict(local_rank=device,
          allocator_fraction=args.memory_fraction, script=args.script)), flush=True)
    try:
        runpy.run_path(str(root / args.script), run_name='__main__')
    finally:
        print('GPU_RESOURCE_PEAK ' + json.dumps(dict(local_rank=device,
              allocated_bytes=torch.cuda.max_memory_allocated(device),
              reserved_bytes=torch.cuda.max_memory_reserved(device))), flush=True)


if __name__ == '__main__':
    main()
