import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_runner_exists():
    assert importlib.util.find_spec('run_project') is not None


def test_f2_builder_accepts_native_instance_options():
    import train
    args = train.parse_args(['--backbone','dinov3','--init','pretrained','--adapt','lora',
                             '--instance-balanced','--native-size','--pretrained-file','init.safetensors',
                             '--accumulation','8'])
    assert args.instance_balanced and args.native_size and args.accumulation == 8
    assert str(args.pretrained_file) == 'init.safetensors'


def test_accumulation_matches_batched_sgd_with_tail(tmp_path):
    import torch
    from types import SimpleNamespace
    from torch import nn
    from run_project import train_epoch
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.fc = nn.Linear(1,9)
        def forward(self, x, targets=None, image_sizes=None):
            return self.fc(x), [], {'det': self.fc.weight.square().mean()*.1}
    torch.manual_seed(13)
    a = Tiny(); b = Tiny(); b.load_state_dict(a.state_dict())
    xs = torch.arange(1,6,dtype=torch.float32)[:,None]/5
    ys = torch.tensor([0,1,2,0,1])
    targets = [{'boxes':torch.zeros(1,4),'labels':torch.ones(1,dtype=torch.long),'valid_size':(1,1)}]
    oa=torch.optim.SGD(a.parameters(),lr=.01); ob=torch.optim.SGD(b.parameters(),lr=.01)
    schedule=torch.optim.lr_scheduler.LambdaLR(oa,lambda _:1)
    loader=[(xs[i:i+1],ys[i:i+1],targets) for i in range(5)]
    args=SimpleNamespace(accumulation=3,cls_weight=1.,label_smoothing=0.)
    step=train_epoch(a,loader,oa,schedule,1,args,tmp_path,0,device=torch.device('cpu'))
    for start in (0,3):
        x=xs[start:start+3]; y=ys[start:start+3]; ob.zero_grad()
        logits,_,loss=b(x)
        (torch.nn.functional.cross_entropy(logits,y)+sum(loss.values())).backward(); ob.step()
    assert step==2
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y)
