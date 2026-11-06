"""The earlier F2 architecture with the new instance-balanced sampling change."""
from pathlib import Path
import train

PRETRAINED = Path('D:/CS701_scoring_correction/pretrained/dinov3_b16/model.safetensors')
PRETRAINED_SHA256 = '1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332'


def arguments(data='D:/CS701_junkai_ib/data', out='D:/CS701_junkai_ib/runs', name='JIB_seed701',
              epochs=100, batch_size=4, accumulation=4):
    return ['--data',str(data),'--out',str(out),'--name',name,'--epochs',str(epochs),
            '--batch-size',str(batch_size),'--accumulation',str(accumulation),'--workers','0','--seed','701',
            '--backbone','dinov3','--init','pretrained','--adapt','lora',
            '--pretrained-file',str(PRETRAINED),'--lora-targets','attn.qkv,attn.proj,mlp.fc1,mlp.fc2',
            '--lora-rank','32','--lora-alpha','64','--train-norms',
            '--backbone-lr','0.0003','--lr','0.0001','--cls-weight','1.0','--weight-decay','0.05',
            '--cascade','--cascade-rep','1024','--cascade-beta','0','--cascade-size-iou','--size-iou-table','a4',
            '--aligned-roi','--rpn-nwd','8','--anchor-offset','0.5',
            '--anchor-sizes','6,8,11;16,19,23;32,38,45;64,76,91;128,152,181',
            '--flip-bias','-0.3','--native-size','--instance-balanced','--train-split','train',
            '--eval-splits','val','--eval-every','4']


def args(**kwargs):
    return train.parse_args(arguments(**kwargs))
