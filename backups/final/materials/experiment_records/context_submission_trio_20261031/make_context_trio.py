from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,os,json,hashlib,zipfile,csv,io,collections,sys
root=Path('/mnt/hdd/junkaichen/CS701/CS701')
sys.path.insert(0,str(root))
from score_submission import check_format
base=Path('/mnt/hdd/junkaichen/CS701/cs701_runs')
record=root/'experiments/context_submission_trio_20261031'
record.mkdir()
old=json.loads((base/'BP7_LR130_local_s0_20261029_v2/lr1x/submission/launch.json').read_text())
trials=[('A_baseline_e150','BP7_fixedLR160_local_s0_20261030_direct/schedule160',150,'BP7_local7512_lr1x_cos160_e150_val_shift-0.2.zip'),
        ('B_context_e150','BP7_ctx3_classification_local160_s0_20261031/context160',150,'BP7_local7512_ctx3_cls_cos160_e150_val_shift-0.2.zip'),
        ('C_context_e130','BP7_ctx3_classification_local160_s0_20261031/context160',130,'BP7_local7512_ctx3_cls_cos160_e130_val_shift-0.2.zip')]
def stamp():return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(record/'plan.json').write_text(json.dumps(dict(created_at=stamp(),trials=trials,training_images=7512,
 schedule_epochs=160,post=old['post'],automatic_upload=False),indent=2)+'\n')
images=json.loads(Path('/mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/val/images.json').read_text())['images']
ids={im['id'] for im in images};sizes={im['id']:(im['width'],im['height']) for im in images}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HF_HOME='/mnt/hdd/junkaichen/hf-cache',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
results={}
for label,relative,epoch,filename in trials:
 run=base/relative;code=run.parent/'code';weights=run/f'ckpt_e{epoch}.pt'
 out=root/'submissions'/filename
 assert not out.exists()
 sub=record/label;sub.mkdir()
 args=json.loads((run/'log.txt').read_text().splitlines()[0])
 assert args['train_split']=='train' and args['schedule_epochs']==160
 assert args['lr']==1e-4 and args['backbone_lr']==3e-4
 if label.startswith('A'):assert args['roi_context']==0
 else:assert args['roi_context']==3 and args['roi_context_mode']=='classification'
 launch=json.loads(json.dumps(old));argv=launch['argv']
 for flag,value in {'--run':str(run),'--weights':weights.name,'--out':str(out),'--raw-out':str(sub/'official_val_raw.pt')}.items():argv[argv.index(flag)+1]=value
 launch.update(started_at=stamp(),source=str(code),run=str(run),checkpoint=str(weights),checkpoint_epoch=epoch,schedule_epochs=160,checkpoint_sha256=digest(weights),output=str(out),roi_context=args['roi_context'],roi_context_mode=args['roi_context_mode'],label=label)
 print('START',label,stamp(),flush=True)
 with (sub/'predict.log').open('w') as log:
  child=subprocess.Popen(argv,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
  launch['pid']=child.pid;(sub/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
  rc=child.wait()
 (sub/'result.json').write_text(json.dumps(dict(returncode=rc,completed_at=stamp()),indent=2)+'\n')
 if rc:print((sub/'predict.log').read_text()[-3000:]);raise SystemExit(rc)
 with zipfile.ZipFile(out) as z:
  assert z.testzip() is None and sorted(z.namelist())==['classification.csv','detection.json']
  raw=z.read('classification.csv').decode();assert raw.splitlines()[0]=='image_id,label'
  rows=list(csv.DictReader(io.StringIO(raw)));dets=json.loads(z.read('detection.json'))
 check_format(rows,dets,ids)
 assert len(rows)==len({r['image_id'] for r in rows})==len(ids)==1426
 counts=collections.Counter(d['image_id'] for d in dets)
 assert set(counts)==ids and max(counts.values())<=100
 outside=sum(not (d['bbox'][0]>=-1e-3 and d['bbox'][1]>=-1e-3 and d['bbox'][0]+d['bbox'][2]<=sizes[d['image_id']][0]+1e-3 and d['bbox'][1]+d['bbox'][3]<=sizes[d['image_id']][1]+1e-3) for d in dets)
 assert outside==0 and out.stat().st_size<25000000
 assert digest(weights)==launch['checkpoint_sha256']
 sha=digest(out);out.with_suffix('.sha256').write_text(sha+'  '+out.name+'\n')
 result=dict(checked_at=stamp(),label=label,path=str(out),checkpoint=str(weights),checkpoint_epoch=epoch,schedule_epochs=160,training_images=7512,images=1426,detections=len(dets),boxes_outside_image=outside,bytes=out.stat().st_size,sha256=sha,checkpoint_sha256=launch['checkpoint_sha256'],format_check_passed=True,crc_passed=True,uploaded=False)
 (sub/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 results[label]=result;(record/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print('COMPLETE',label,result['bytes'],result['detections'],flush=True)
print('ALL_COMPLETE',stamp(),flush=True)
