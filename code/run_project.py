"""Run the independent F2-based model with accumulation and full epoch-boundary recovery.

This entry imports only the copied F2 project and standard research libraries.
It does not import the previous local project's model or training implementation.
"""
import contextlib
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train as f2
from run_safety import (atomic_json, check_resume, data_fingerprint, group_length,
                        internal_delta, sha256, source_hashes)
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import classification_metrics, detection_metrics
from native_postprocessing import FINAL_POST, fused_class


def emit(out, row):
    row = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **row}
    line = json.dumps(row, default=str)
    print(line, flush=True)
    with (Path(out)/'progress.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')


def train_epoch(model, loader, optimizer, schedule, epoch, args, out, step0, device=None):
    device = device or torch.device('cuda')
    model.train(); optimizer.zero_grad(set_to_none=True)
    begin = time.perf_counter(); updates = step0
    for index, (images, labels, targets) in enumerate(loader):
        boxes = [{'boxes':t['boxes'].to(device), 'labels':t['labels'].to(device)} for t in targets]
        sizes = [t['valid_size'] for t in targets]
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
        with autocast:
            logits, _, losses = model(images.to(device), targets=boxes, image_sizes=sizes)
            loss = args.cls_weight * F.cross_entropy(logits.float(), labels.to(device),
                                                     label_smoothing=args.label_smoothing) + sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at epoch {epoch}, microbatch {index+1}')
        (loss / group_length(index, len(loader), args.accumulation)).backward()
        if (index+1) % args.accumulation == 0 or index+1 == len(loader):
            optimizer.step(); schedule.step(); optimizer.zero_grad(set_to_none=True); updates += 1
            if updates % 25 == 0 or index+1 == len(loader):
                emit(out, {'status':'TRAINING', 'epoch':epoch, 'microbatch':index+1,
                           'microbatches_per_epoch':len(loader), 'optimizer_update':updates,
                           'loss':float(loss.detach()), 'epoch_seconds':round(time.perf_counter()-begin,1),
                           'learning_rates':[g['lr'] for g in optimizer.param_groups],
                           'gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if device.type=='cuda' else 0})
    return updates


@torch.no_grad()
def evaluate_native(model, loader, gt_json, device):
    model.eval(); truth=[]; predicted=[]; ids=[]; detections=[]
    model.roi_heads.post.update(FINAL_POST)
    for images, labels, targets in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, outputs, _ = model(images.to(device), image_sizes=[t['valid_size'] for t in targets], return_raw=True)
        truth.extend(labels.tolist()); predicted.extend(fused_class(lg,out) for lg,out in zip(logits,outputs))
        for target, output in zip(targets,outputs):
            image_id=int(target['image_id']); ids.append(image_id)
            boxes=to_original_xywh(output['boxes'], target['orig_size'],
                                   valid_size=target['valid_size'], scale_xy=target['scale_xy']).tolist()
            for box, label, score in zip(boxes, output['labels'].tolist(),output['scores'].tolist()):
                if box[2] > 0 and box[3] > 0:
                    detections.append({'image_id':image_id,'category_id':int(label),'bbox':box,'score':float(score)})
    metrics={**classification_metrics(truth,predicted),**detection_metrics(str(gt_json),detections,ids)}
    metrics['internal_delta_m_percent']=internal_delta(metrics['macro_f1'],metrics['mAP'])
    metrics['official_delta_m']=None
    return metrics,detections


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, payload):
    path=Path(path); tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp); os.replace(tmp,path)


def main():
    args=f2.parse_args(); root=Path(__file__).resolve().parent
    if not args.native_size or args.backbone!='dinov3' or not args.cascade or not args.instance_balanced:
        raise ValueError('This run requires native DINOv3, cascade and instance-balanced sampling')
    if args.accumulation<1 or args.batch_size<1 or args.batch_size*args.accumulation!=16:
        raise ValueError('Use positive microbatch and accumulation, effective batch16')
    if args.workers!=0 or args.compile or args.ema or args.copy_paste or args.zoom_p or args.limit:
        raise ValueError('Use workers0, eager execution, no EMA/copy-paste/zoom/limit for this audited run')
    if args.eval_splits!='val' or args.train_split!='train':
        raise ValueError('The current run uses the fixed internal train/val partition')
    if not args.name:raise ValueError('Provide --name for an auditable standalone run')
    out=args.out/args.name
    if not args.resume:
        if out.exists() and any(out.iterdir()):raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True)
    elif args.resume.resolve()!=(out/'last_checkpoint.pt').resolve():
        raise ValueError('Resume must use the same run latest completed-epoch checkpoint')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    # F2 enabled TF32 indirectly through a TerraTorch import. Keep that setting explicit for DINO only.
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    emit(out,{'status':'VERIFYING_DATA','model':'Junkai F2 + instance-balanced cascade','epochs':args.epochs})
    fingerprint=data_fingerprint(args.data)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(data_fingerprint=fingerprint, source_hashes=source_hashes(root),
                  pretrained_sha256=sha256(args.pretrained_file), torch_version=torch.__version__,
                  cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  evaluation_postprocessing=FINAL_POST, classification_fusion_lambda=1.,
                  selection='maximum internal Delta_m over planned evaluated epochs; official score unknown')
    previous=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if previous:
        check_resume(previous['config'],config)
        if not previous['at_epoch_boundary']:raise ValueError('Not an epoch-boundary checkpoint')
    else:atomic_json(out/'config.json',config)
    train_data=SARMultiTask(args.data,'train',train=True,aug=args.aug,flip_bias=args.flip_bias,
                           native_size=True,mean=args.mean,std=args.std)
    val_data=SARMultiTask(args.data,'val',native_size=True,mean=args.mean,std=args.std)
    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=0,
                            collate_fn=collate,pin_memory=True)
    val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=0,
                          collate_fn=collate,pin_memory=True)
    if len(train_data) % (args.batch_size*args.accumulation):
        raise ValueError('Audited run requires complete effective batches; re-plan sample weighting for other data counts')
    build_args=copy.copy(args)
    if previous:build_args.init='scratch'
    model=f2.build_model(build_args)
    if previous:model.load_state_dict(previous['model'],strict=True)
    parameters=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers=sum(b.numel() for b in model.buffers() if b.is_floating_point())
    if parameters+buffers>130_000_000 or trainable>40_000_000:raise RuntimeError('Parameter cap exceeded')
    model.cuda(); device=torch.device('cuda')
    optimizer=torch.optim.AdamW(f2.param_groups(model,args),lr=args.lr,weight_decay=args.weight_decay)
    updates_per_epoch=math.ceil(len(train_loader)/args.accumulation)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:f2.lr_factor(step,updates_per_epoch,
                                                                                 args.epochs*updates_per_epoch))
    history=[]; completed=0; updates=0; best=-float('inf'); elapsed=0.; best_epoch=0; best_sha=None
    if previous:
        optimizer.load_state_dict(previous['optimizer']);scheduler.load_state_dict(previous['scheduler'])
        history=previous['history'];completed=previous['completed_epoch'];updates=previous['updates']
        best=previous['best_internal_delta'];elapsed=previous['elapsed_seconds'];restore_rng(previous['rng'])
        best_epoch=previous['best_epoch'];best_sha=previous['best_checkpoint_sha256']
        if best_epoch:
            best_path=out/'best_epochs'/f'epoch_{best_epoch:04d}.pt'
            assert sha256(best_path)==best_sha
            atomic_json(out/'best_selection.json',{'status':'PROVISIONAL','selected_epoch':best_epoch,
                        'internal_delta_m_percent':best,'official_delta_m':None,
                        'checkpoint_file':str(best_path),'checkpoint_sha256':best_sha,
                        'scope':'best among planned evaluated epochs, not all training epochs'})
        assert updates==completed*updates_per_epoch
        del previous
    emit(out,{'status':'READY','parameters':parameters,'trainable_parameters':trainable,'buffers':buffers,
               'train_images':len(train_data),'val_images':len(val_data),'updates_per_epoch':updates_per_epoch,
               'completed_epochs':completed,'instance_balanced':True})
    start=time.perf_counter()
    for epoch in range(completed+1,args.epochs+1):
        updates=train_epoch(model,train_loader,optimizer,scheduler,epoch,args,out,updates,device)
        improved=False
        if epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs:
            metrics,detections=evaluate_native(model,val_loader,args.data/'detection/instances_val.json',device)
            row={'epoch':epoch,'optimizer_updates':updates,**metrics};history.append(row)
            improved=metrics['internal_delta_m_percent']>best
            if improved:best=metrics['internal_delta_m_percent']
            if improved:best_epoch=epoch
            atomic_json(out/'evaluation_history.json',history)
            if improved:atomic_json(out/'best_predictions_val.json',detections)
            emit(out,{'status':'VALIDATED','is_best':improved,**row})
        payload={'format':'junkai_ib_full_v1','completed_epoch':epoch,'at_epoch_boundary':True,
                 'updates':updates,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scheduler':scheduler.state_dict(),'rng':capture_rng(),'history':history,
                 'best_internal_delta':best,'config':config,'elapsed_seconds':elapsed+time.perf_counter()-start,
                 'best_epoch':best_epoch,'best_checkpoint_sha256':best_sha,
                 'parameters':parameters,'trainable_parameters':trainable}
        if improved:
            (out/'best_epochs').mkdir(exist_ok=True)
            best_path=out/'best_epochs'/f'epoch_{epoch:04d}.pt'
            save_checkpoint(best_path,payload)
            best_sha=sha256(best_path)
            payload['best_checkpoint_sha256']=best_sha
        # Commit the epoch only after its improved weights are retained. Earlier best epochs remain available.
        save_checkpoint(out/'last_checkpoint.pt',payload)
        if improved:
            atomic_json(out/'best_selection.json',{'status':'PROVISIONAL' if epoch<args.epochs else 'PENDING_RUN_FINALIZATION',
                        'selected_epoch':epoch,'internal_delta_m_percent':best,'official_delta_m':None,
                        'checkpoint_file':str(best_path),'checkpoint_sha256':best_sha,
                        'scope':'best among planned evaluated epochs, not all training epochs'})
        emit(out,{'status':'EPOCH_SAVED','epoch':epoch,'updates':updates,
                   'elapsed_seconds':round(elapsed+time.perf_counter()-start,1)})
        del payload
    summary={'status':'COMPLETED','completed_epochs':args.epochs,'optimizer_updates':updates,
             'elapsed_seconds':elapsed+time.perf_counter()-start,'best_internal_delta_m_percent':best,
             'planned_evaluated_epochs':[1]+[i for i in range(2,args.epochs+1) if i%args.eval_every==0 or i==args.epochs],
             'evaluated_epochs':[x['epoch'] for x in history],'official_delta_m':None}
    assert summary['planned_evaluated_epochs']==summary['evaluated_epochs']
    atomic_json(out/'run_summary.json',summary)
    selection=json.loads((out/'best_selection.json').read_text(encoding='utf-8'))
    assert sha256(selection['checkpoint_file'])==selection['checkpoint_sha256']
    selection.update(status='COMPLETED',completed_epochs=args.epochs,
                     verified_over_planned_evaluated_epochs=True)
    atomic_json(out/'best_selection.json',selection);emit(out,summary)


if __name__=='__main__':main()
