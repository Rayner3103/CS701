"""Equal parameter average of ckpt_e{E}.pt files in one run dir (model.pt = the final epoch given by --final-epoch)."""
import argparse, hashlib, json
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    ap.add_argument('--final-epoch', type=int, help='epoch number that model.pt corresponds to')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists(), a.out
    states, prov = [], []
    for e in a.epochs:
        p = a.dir / ('model.pt' if a.final_epoch == e else f'ckpt_e{e}.pt')
        assert p.exists(), p
        s = torch.load(p, map_location='cpu', weights_only=True)
        if states:
            assert s.keys() == states[0].keys() and all(t.shape == states[0][k].shape for k, t in s.items())
        states.append(s); prov.append(dict(epoch=e, path=str(p)))
    avg = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float64)
        for s in states:
            acc.add_(s[k].double())
        avg[k] = acc.div_(len(states)).to(torch.float32)
        assert torch.isfinite(avg[k]).all()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg, a.out)
    a.out.with_suffix('.json').write_text(json.dumps(dict(sources=prov, tensors=len(avg)), indent=2))
    print('wrote', a.out, len(states), 'checkpoints')

if __name__ == '__main__':
    main()
