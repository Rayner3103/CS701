"""Epoch-boundary training checkpoints, separate from inference-only model.pt files."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

FORMAT_VERSION = 1


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def seed_loader_epoch(loader, seed, epoch, rank=0):
    """Workers are restarted each epoch; their seeds do not depend on earlier iterations."""
    value = seed + epoch * 100_003 + rank * 1_009
    if loader.generator is not None:
        loader.generator.manual_seed(value)
    sampler_generator = getattr(loader.sampler, 'generator', None)
    if sampler_generator is not None and sampler_generator is not loader.generator:
        sampler_generator.manual_seed(value)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def canonical_name(name):
    return name.replace('._orig_mod', '')


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_torch_save(value, destination):
    """A failed write leaves the previous checkpoint intact."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def compatibility_signature(args, model, loader, world):
    ignored = {'out', 'name', 'resume', 'stop_after_epoch', 'save_training_state', 'save', 'save_every',
               'dump_raw', 'allow_over_budget'}
    config = {k: v for k, v in vars(args).items() if k not in ignored}
    # Normalize Path values and tuples to the same JSON types on save and load.
    config = json.loads(json.dumps(config, default=str))
    manifests = {}
    splits = set(args.train_split.split('+')) | set(s for s in args.eval_splits.split(',') if s)
    for split in sorted(splits):
        for rel in (f'classification/{split}.csv', f'detection/instances_{split}.json'):
            manifests[rel] = hashlib.sha256((args.data / rel).read_bytes()).hexdigest()
    return {'args': config, 'world_size': world, 'steps_per_epoch': len(loader),
            'dataset_size': len(loader.dataset), 'sampler': type(loader.sampler).__name__,
            'manifests': manifests,
            'parameters': {canonical_name(n): {'shape': list(p.shape), 'trainable': p.requires_grad}
                           for n, p in model.named_parameters()}}


def validate_compatibility(saved, current):
    if saved != current:
        differences = []
        for key in sorted(set(saved) | set(current)):
            if saved.get(key) != current.get(key):
                if key == 'args':
                    a, b = saved.get(key, {}), current.get(key, {})
                    differences.extend(f'args.{k}: {a.get(k)!r} != {b.get(k)!r}'
                                       for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k))
                else:
                    differences.append(key)
        raise ValueError('Resume configuration mismatch: ' + '; '.join(differences))


def save_training_state(path, model, optimizer, schedule, ema, *, epoch, step, history,
                        train_hours, signature, world=1, rank=0):
    local_rng = rng_state()
    if world > 1:
        rank_rng = [None] * world
        dist.all_gather_object(rank_rng, local_rng)
    else:
        rank_rng = [local_rng]
    if rank == 0:
        state = {'format_version': FORMAT_VERSION,
                 'model': {canonical_name(k): v.detach().cpu() for k, v in model.state_dict().items()},
                 'optimizer': cpu_tree(optimizer.state_dict()), 'scheduler': schedule.state_dict(),
                 'ema': None if ema is None else {'decay': ema.decay, 'shadow': cpu_tree(ema.shadow)},
                 'epoch': epoch, 'global_step': step, 'history': history, 'train_hours': train_hours,
                 'signature': signature, 'rank_rng': rank_rng}
        atomic_torch_save(state, path)
    if world > 1:
        dist.barrier()


def load_training_state(path, model, optimizer, schedule, ema, *, signature, rank=0):
    # --resume is a trusted local training checkpoint, not an untrusted uploaded model.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported training checkpoint format (model.pt is inference-only)')
    validate_compatibility(state['signature'], signature)
    expected_step = state['epoch'] * signature['steps_per_epoch']
    if state['global_step'] != expected_step or state['scheduler']['last_epoch'] != expected_step:
        raise ValueError('Checkpoint epoch, global step and scheduler step do not agree')
    names = {canonical_name(k): k for k in model.state_dict()}
    model.load_state_dict({names.get(k, k): v for k, v in state['model'].items()}, strict=True)
    optimizer.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['scheduler'])
    if (ema is None) != (state['ema'] is None):
        raise ValueError('Checkpoint EMA does not match the current configuration')
    if ema is not None:
        if ema.decay != state['ema']['decay'] or len(ema.shadow) != len(state['ema']['shadow']):
            raise ValueError('Checkpoint EMA configuration mismatch')
        for target, source in zip(ema.shadow, state['ema']['shadow']):
            target.copy_(source.to(target.device))
    restore_rng(state['rank_rng'][rank])
    # Drop the large tensor trees after restoration; retain just loop progress.
    return {k: state[k] for k in ('epoch', 'global_step', 'history', 'train_hours')}
