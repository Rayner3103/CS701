"""Bounded real-data checks for the independent F2-based model. No accuracy claims."""
import copy
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import train
from recipe import args as recipe_args, PRETRAINED, PRETRAINED_SHA256
from run_safety import atomic_json, sha256
from sarbench.data import SARMultiTask, collate


def main():
    torch.set_num_threads(4); torch.manual_seed(701)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,choices=(2,4),default=4)
    opts=parser.parse_args()
    args=recipe_args(batch_size=opts.batch_size,accumulation=16//opts.batch_size)
    assert sha256(PRETRAINED)==PRETRAINED_SHA256
    dataset=SARMultiTask(args.data,'train',native_size=True,mean=args.mean,std=args.std)
    indexes=[]
    for size in (256,512):
        indexes.extend([i for i,row in enumerate(dataset.rows)
                            if dataset.coco_images[row['image']]['width']==size
                            and dataset.coco_images[row['image']]['height']==size][:opts.batch_size//2])
    images,labels,targets=collate([dataset[i] for i in indexes]); sizes=[t['valid_size'] for t in targets]
    images,labels=images.cuda(),labels.cuda()
    boxes=[{'boxes':t['boxes'].cuda(),'labels':t['labels'].cuda()} for t in targets]
    model=train.build_model(args).cuda()
    counts={'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'buffers':sum(b.numel() for b in model.buffers() if b.is_floating_point())}
    assert counts['total']+counts['buffers']<=130_000_000 and counts['trainable']<=40_000_000
    optimizer=torch.optim.AdamW(train.param_groups(model,args),lr=args.lr,weight_decay=args.weight_decay)
    torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();loss_values=[]
    for _ in range(2):
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_,losses=model(images,targets=boxes,image_sizes=sizes)
            loss=F.cross_entropy(logits.float(),labels)+sum(losses.values())
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        for head in model.roi_heads.stage_preds:
            assert head.cls_score.weight.grad is not None and head.cls_score.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step();loss_values.append(float(loss.detach()))
    worst=sorted(range(len(dataset)),key=lambda i:len(dataset.annotations[dataset.coco_images[dataset.rows[i]['image']]['id']]),
                 reverse=True)[:opts.batch_size]
    wi,wl,wt=collate([dataset[i] for i in worst]);wi,wl=wi.cuda(),wl.cuda()
    wboxes=[{'boxes':t['boxes'].cuda(),'labels':t['labels'].cuda()} for t in wt]
    model.train();optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        wz,_,wlosses=model(wi,targets=wboxes,image_sizes=[t['valid_size'] for t in wt])
        wloss=F.cross_entropy(wz.float(),wl)+sum(wlosses.values())
    assert torch.isfinite(wloss);wloss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected,detections,_=model(images,image_sizes=sizes)
    expected=expected.cpu(); expected_dets=[{k:d[k].cpu() for k in ('boxes','scores','labels')} for d in detections]
    for d,(h,w) in zip(expected_dets,sizes):
        assert len(d['boxes'])<=100 and torch.isfinite(d['boxes']).all()
        assert (d['boxes'][:,0::2]<=w).all() and (d['boxes'][:,1::2]<=h).all()
    memory={'allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    file=Path('D:/CS701_junkai_ib/qualification_checkpoint.pt');file.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'args':vars(args),
                'config':vars(args),'completed_epoch':0},file)
    del model,optimizer,losses,loss,logits,detections,wi,wl,wboxes,wz,wlosses,wloss
    gc.collect();torch.cuda.empty_cache()
    payload=torch.load(file,map_location='cpu',weights_only=False)
    restore_args=copy.copy(args);restore_args.init='scratch'
    restored=train.build_model(restore_args);restored.load_state_dict(payload['model'],strict=True);restored.cuda().eval()
    restored_optimizer=torch.optim.AdamW(train.param_groups(restored,args),lr=args.lr,weight_decay=args.weight_decay)
    restored_optimizer.load_state_dict(payload['optimizer']);assert restored_optimizer.state
    del payload
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        actual,actual_dets,_=restored(images,image_sizes=sizes)
    torch.testing.assert_close(actual.cpu(),expected,atol=1e-5,rtol=1e-5)
    for actual_det,expected_det in zip(actual_dets,expected_dets):
        for k in ('boxes','scores','labels'):
            torch.testing.assert_close(actual_det[k].cpu(),expected_det[k],atol=1e-5,rtol=1e-5)
    result={'status':'PASSED','scope':'two optimizer steps, worst-GT batch backward, full model/optimizer restoration; no accuracy evidence',
            'model':'Junkai F2 + instance-balanced Cascade, native-size input','counts':counts,'microbatch':opts.batch_size,
            'mixed_valid_sizes':sizes,'losses':loss_values,'worst_gt_counts':[len(t['boxes']) for t in wt],
            'peak_memory':memory,'elapsed_seconds':time.perf_counter()-begin,'pretrained_sha256':PRETRAINED_SHA256,
            'trained_old_model_weights_used':False,'sc2_configuration_used':False}
    atomic_json(Path('D:/CS701_junkai_ib/qualification.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
