"""Equal (or weighted) parameter average of BP7 cos200 checkpoints -> one single-model weights file.

Usage:
  python average_ckpts.py --epochs 100 110 120 130 140 150 160 170 180 --out /path/avg_e100-180.pt
Only averages trained tensors (the 234 saved tensors); same keys/shapes are asserted. CPU only.
"""
import argparse, hashlib, json
from pathlib import Path
import torch

RUNS = Path('/mnt/hdd/junkaichen/CS701/cs701_runs')
CKPT = {}
for e in range(10, 131, 10):
    CKPT[e] = RUNS / f'BP7_full9392_cos200_stop160_s0_20261031_expandable/full_train/ckpt_e{e}.pt'
CKPT[140] = RUNS / 'BP7_full9392_cos200_e130to160_s0_20261101_mem35/full_train_e160/ckpt_e140.pt'
CKPT[150] = RUNS / 'BP7_full9392_cos200_e147to160_s0_20261101_tmp/full_train_e160/ckpt_e150.pt'
CKPT[160] = RUNS / 'BP7_full9392_cos200_e147to160_s0_20261101_tmp/full_train_e160/ckpt_e160.pt'
for e in (170, 180, 190):
    CKPT[e] = RUNS / f'BP7_full9392_cos200_e160to200_s0_20261101_v2/full_train_e200/ckpt_e{e}.pt'
CKPT[200] = RUNS / 'BP7_full9392_cos200_e160to200_s0_20261101_v2/full_train_e200/model.pt'

def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--weights', type=float, nargs='*', help='optional per-checkpoint weights (default equal)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists(), f'{a.out} exists; refuse to overwrite'
    w = a.weights or [1.0] * len(a.epochs)
    assert len(w) == len(a.epochs)
    w = [x / sum(w) for x in w]
    states, prov = [], []
    for e, c in zip(a.epochs, w):
        p = CKPT[e]
        assert p.exists(), p
        s = torch.load(p, map_location='cpu', weights_only=True)
        if states:
            assert s.keys() == states[0].keys()
            assert all(t.shape == states[0][k].shape for k, t in s.items())
        states.append(s)
        prov.append(dict(epoch=e, path=str(p), sha256=sha(p), coefficient=c))
    avg = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float64)
        for s, c in zip(states, w):
            acc.add_(s[k].double() * c)
        avg[k] = acc.to(torch.float32)
        assert torch.isfinite(avg[k]).all()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg, a.out)
    meta = dict(method='parameter average, one model, one forward per image', sources=prov,
                tensor_count=len(avg), values=sum(t.numel() for t in avg.values()), output_sha256=sha(a.out))
    a.out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
