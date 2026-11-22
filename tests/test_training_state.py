"""Checkpoint correctness with stochastic CPU training; no backbone download or GPU required."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sarbench.training_state import (atomic_torch_save, compatibility_signature, load_training_state,
                                     save_training_state, seed_everything, seed_loader_epoch, seed_worker,
                                     validate_compatibility)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(3, 3).requires_grad_(False)
        self.block = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(0.2), nn.Linear(3, 1))

    def forward(self, x):
        return self.block(self.frozen(x))


class TinyEMA:
    def __init__(self, model):
        self.decay = 0.9
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        for value, param in zip(self.shadow, self.params):
            value.lerp_(param, 1 - self.decay)


def setup(use_ema=False):
    model = TinyModel()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (8 - step) / 8)
    ema = TinyEMA(model) if use_ema else None
    return model, optimizer, scheduler, ema


def advance(parts, steps):
    model, optimizer, scheduler, ema = parts
    losses = []
    for _ in range(steps):
        x = torch.randn(4, 3) + random.random() + float(np.random.uniform())
        y = torch.randn(4, 1)
        loss = (model(x) - y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if ema:
            ema.update()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize('use_ema', [False, True])
def test_resume_matches_uninterrupted_stochastic_training(tmp_path, use_ema):
    seed_everything(17)
    original = setup(use_ema)
    advance(original, 2)
    # Validation can consume RNG: capture after it, not before it.
    torch.rand(5)
    random.random()
    np.random.random()
    signature = {'steps_per_epoch': 2, 'world_size': 1, 'args': {'lr': 0.01}}
    checkpoint = tmp_path / 'training_state.pt'
    save_training_state(checkpoint, *original, epoch=1, step=2, history=[{'epoch': 1, 'mAP': 0.4}],
                        train_hours=0.2, signature=signature)
    expected_losses = advance(original, 4)
    expected_model = {k: v.clone() for k, v in original[0].state_dict().items()}
    # Construction deliberately changes all RNG and parameter values before restoring.
    seed_everything(999)
    restarted = setup(use_ema)
    progress = load_training_state(checkpoint, *restarted, signature=signature)
    assert progress == {'epoch': 1, 'global_step': 2, 'history': [{'epoch': 1, 'mAP': 0.4}], 'train_hours': 0.2}
    assert advance(restarted, 4) == expected_losses
    for name, value in restarted[0].state_dict().items():
        assert torch.equal(value, expected_model[name]), name
    assert restarted[2].state_dict() == original[2].state_dict()
    for p, q in zip(original[1].state.values(), restarted[1].state.values()):
        for key in p:
            assert torch.equal(p[key], q[key])
    if use_ema:
        assert all(torch.equal(a, b) for a, b in zip(original[3].shadow, restarted[3].shadow))


def test_failed_write_keeps_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'training_state.pt'
    atomic_torch_save({'epoch': 1}, path)
    before = path.read_bytes()

    def fail(_value, stream):
        stream.write(b'partial broken checkpoint')
        raise OSError('disk full')

    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        atomic_torch_save({'epoch': 2}, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_config_mismatch_reports_changed_lr():
    with pytest.raises(ValueError, match=r'args.lr'):
        validate_compatibility({'args': {'lr': 0.01}}, {'args': {'lr': 0.02}})


def test_config_tracks_data_and_trainable_parameters(tmp_path):
    (tmp_path / 'classification').mkdir()
    (tmp_path / 'detection').mkdir()
    csv_path = tmp_path / 'classification/train.csv'
    csv_path.write_text('image,label\na.bmp,0\n')
    (tmp_path / 'detection/instances_train.json').write_text('{}')
    args = SimpleNamespace(data=tmp_path, train_split='train', eval_splits='', lr=0.01,
                           out=tmp_path / 'out', resume=None)
    model = TinyModel()
    loader = DataLoader(torch.ones(4, 3), batch_size=2)
    signature = compatibility_signature(args, model, loader, 1)
    args.out = tmp_path / 'new_out'
    args.resume = tmp_path / 'state.pt'
    assert signature == compatibility_signature(args, model, loader, 1)
    csv_path.write_text('image,label\nb.bmp,0\n')
    with pytest.raises(ValueError, match='manifests'):
        validate_compatibility(signature, compatibility_signature(args, model, loader, 1))


class StochasticDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), np.random.random(), torch.rand(()).item()])


def loader():
    return DataLoader(StochasticDataset(), batch_size=3, shuffle=True, num_workers=2,
                      generator=torch.Generator(), worker_init_fn=seed_worker, persistent_workers=False)


def test_restarted_workers_match_next_epoch():
    uninterrupted = loader()
    seed_loader_epoch(uninterrupted, 3, 1)
    list(uninterrupted)
    seed_loader_epoch(uninterrupted, 3, 2)
    expected = torch.cat(list(uninterrupted))
    restarted = loader()
    seed_loader_epoch(restarted, 3, 2)
    assert torch.equal(expected, torch.cat(list(restarted)))


def test_compiled_name_mapping(tmp_path):
    parts = setup()
    advance(parts, 2)
    signature = {'steps_per_epoch': 2}
    path = tmp_path / 'training_state.pt'
    save_training_state(path, *parts, epoch=1, step=2, history=[], train_hours=0, signature=signature)
    restored = setup()

    class CompileWrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

        def forward(self, x):
            return self._orig_mod(x)

    restored[0].block = CompileWrapper(restored[0].block)
    load_training_state(path, *restored, signature=signature)
    for name, value in restored[0].state_dict().items():
        assert torch.equal(value, parts[0].state_dict()[name.replace('._orig_mod', '')])
