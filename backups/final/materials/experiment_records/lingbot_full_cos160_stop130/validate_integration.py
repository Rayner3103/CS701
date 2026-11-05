"""CPU-only actual-weight parity and gradient checks for the LingBot adapter."""
from pathlib import Path
import os
import sys
import json

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS,0)))
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from huggingface_hub import hf_hub_download
from sarbench.lingbot import LingBotViT, base_config, HF_REVISION, HF_REPO
from sarbench.vendor.lingbot_vision.vit import LingBotVisionTransformer
from sarbench.adapters import add_lora, unfreeze

torch.set_num_threads(1)
torch.manual_seed(0)
path=hf_hub_download(repo_id=HF_REPO,filename='model.pt',revision=HF_REVISION)
state=torch.load(path,map_location='cpu',weights_only=True)['model']
rgb=LingBotVisionTransformer(**base_config(64,3))
rgb.load_state_dict(state,strict=True);rgb.eval()
gray=LingBotViT(True,img_size=64).eval()
x=torch.randn(2,1,64,64)
with torch.no_grad():
    ref=rgb.forward_features(x.repeat(1,3,1,1))['x_norm_patchtokens']
    actual=gray(x)
diff=(ref-actual).abs().max().item()
print('GRAY_PARITY',diff,'SHAPE',tuple(actual.shape),'FINITE',bool(torch.isfinite(actual).all()),flush=True)
torch.testing.assert_close(actual,ref,rtol=3e-4,atol=1e-4)
assert gray.prefix_tokens.shape==(2,5,768)
del rgb,state,ref,actual
wrapped=add_lora(gray,('attn.qkv','attn.proj','mlp.fc1','mlp.fc2'),80,160)
unfreeze(gray,norms=True);gray.train()
probe=torch.randn(2,16,768)
def run(checkpointing):
    gray.zero_grad(set_to_none=True);gray.vit.set_grad_checkpointing(checkpointing)
    out=gray(x);loss=(out*probe).mean();loss.backward()
    grad={n:p.grad.clone() for n,p in gray.named_parameters() if p.requires_grad and p.grad is not None}
    assert len(grad)==sum(p.requires_grad for p in gray.parameters())
    assert all(torch.isfinite(g).all() for g in grad.values())
    return out.detach(),grad
plain,ga=run(False);checked,gb=run(True)
torch.testing.assert_close(plain,checked,rtol=0,atol=0)
for k in ga:torch.testing.assert_close(ga[k],gb[k],rtol=1e-5,atol=1e-7)
gradmax=max((ga[k]-gb[k]).abs().max().item() for k in ga)
print('GRAD_CHECKPOINT_PARITY',gradmax,'TRAINED_TENSORS',len(ga),'LORA_LAYERS',wrapped,flush=True)
# Check Dynamo can trace the upstream block through LoRA and checkpointing,
# without invoking GPU or spending time compiling every CPU linear kernel.
gray.blocks[0]=torch.compile(gray.blocks[0],backend='eager',fullgraph=True)
compiled,gc=run(True)
torch.testing.assert_close(plain,compiled,rtol=0,atol=0)
canonical={k.replace('._orig_mod',''):v for k,v in gc.items()}
for k in ga:torch.testing.assert_close(ga[k],canonical[k],rtol=1e-5,atol=1e-7)
print('DYNAMO_FULLGRAPH_BLOCK_CHECKPOINT_PASSED',flush=True)
del gray,ga,gb,gc
import train
saved=json.loads((ROOT.parent/'cs701_runs/BP7_full9392_cos160_stop130_s0_20261031/full_train/log.txt').read_text().splitlines()[0])
args=train.parse_args(['--backbone','lingbot_b','--init','pretrained','--adapt','lora'])
for k,v in saved.items():setattr(args,k,v)
args.backbone='lingbot_b';args.init='scratch'
with torch.device('meta'):model=train.build_model(args)
count=dict(parameters=sum(p.numel() for p in model.parameters()),
           trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
           floating_buffers=sum(b.numel() for b in model.buffers() if b.is_floating_point()),
           all_buffers=sum(b.numel() for b in model.buffers()))
assert count['parameters']+count['floating_buffers']<=130000000 and count['trainable_parameters']<=40000000
print('BUDGET',count,flush=True)
result=dict(cpu_only=True,weights_path=path,strict_load=True,gray_rgb_token_max_abs_diff=diff,
            input_shape=[2,1,64,64],grad_checkpoint_max_abs_diff=gradmax,gradients_finite=True,
            dynamo_fullgraph_block_with_checkpoint=True,lora_wrapped_layers=wrapped,budget=count,
            limitations='CPU eager-backend Dynamo test does not replace the queued CUDA/Inductor DDP smoke.')
(Path(__file__).parent/'cpu_integration_validation.json').write_text(json.dumps(result,indent=2)+'\n')
