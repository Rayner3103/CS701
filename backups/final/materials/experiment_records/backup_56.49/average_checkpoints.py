"""Equal parameter average of checkpoints -> one single-model weights file (the 56.49 model).

  python average_checkpoints.py                                  # weights/ckpt_e100..e180.pt -> weights/model.pt
  python average_checkpoints.py --dir runs/BP7_cos200 --out my_avg.pt   # a retrained trajectory
  python average_checkpoints.py --epochs 100 120 140 160          # another window

Only the trained tensors are averaged (234 tensors, 39,867,360 values; the frozen DINOv3 weights are shared).
Accumulates in float64 as c * tensor, c = 1/n, then stores float32: with the backed-up checkpoints this reproduces
weights/model.pt bit for bit. Writes <out>.json with every source's SHA256. CPU only, about 30 s.
"""
import argparse, hashlib, json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent

def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, default=HERE / 'weights', help='folder holding ckpt_e<E>.pt files')
    ap.add_argument('--epochs', type=int, nargs='+', default=list(range(100, 181, 10)))
    ap.add_argument('--out', type=Path, default=HERE / 'weights' / 'model.pt')
    a = ap.parse_args()
    assert not a.out.exists(), f'{a.out} exists; refusing to overwrite'
    c = 1.0 / len(a.epochs)
    states, prov = [], []
    for e in a.epochs:
        p = a.dir / f'ckpt_e{e}.pt'
        assert p.exists(), p
        s = torch.load(p, map_location='cpu', weights_only=True)
        if states:
            assert s.keys() == states[0].keys() and all(t.shape == states[0][k].shape for k, t in s.items())
        states.append(s)
        prov.append(dict(epoch=e, path=str(p), sha256=sha(p), coefficient=c))
    avg = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float64)
        for s in states:
            acc.add_(s[k].double() * c)
        avg[k] = acc.to(torch.float32)
        assert torch.isfinite(avg[k]).all()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg, a.out)
    meta = dict(method='equal parameter average; one model, one forward pass per image', sources=prov,
                tensor_count=len(avg), values=sum(t.numel() for t in avg.values()), output_sha256=sha(a.out))
    a.out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
