"""Average the weights of several checkpoints of ONE training run into a single model (uniform weight average,
no ensembling: the result is one network evaluated once per image). Non-float tensors are taken from the first.
  python avg_checkpoints.py OUT.pt ckpt_e120.pt ckpt_e130.pt ckpt_e140.pt"""
import sys

import torch

out, ins = sys.argv[1], sys.argv[2:]
states = [torch.load(p, map_location='cpu') for p in ins]
avg = {}
for k, v in states[0].items():
    avg[k] = (sum(s[k].double() for s in states) / len(states)).to(v.dtype) if v.is_floating_point() else v
torch.save(avg, out)
print(len(avg), 'tensors averaged from', len(ins))
