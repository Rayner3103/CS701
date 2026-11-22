import os,sys,json,zipfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path('/mnt/hdd/junkaichen/CS701/CS701')
suite=root.parent/'cs701_runs/BP7_full9392_cos200_e160to200_s0_20261101_v2'
work=root/'experiments/bp7_full_cos200_final/e180_softsigma05'
sys.path[:0]=[str(suite/'code'),str(suite/'code/tools')]
import torch
import predict
from run_bp7_cos200_final import validate_zip
from run_bp7_e130_recovery import verify_frozen
from run_bp7_schedule_search import write_json,sha256_file
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
if os.getpriority(os.PRIO_PROCESS,0)<10: os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
verify_frozen(suite)
raw=root/'experiments/bp7_full_cos200_final/e180_immediate/raw.pt'
base=root/'submissions/BP7_e180_cls2.zip'
out=root/'submissions/BP7_e180_soft05_cls2.zip'
assert not out.exists() and not out.with_suffix('.sha256').exists()
assert sha256_file(raw)=='b6241b1dda6f2ce601781e1aeaa6970559292e3f38c680dca1b11022d03c691c'
assert sha256_file(base)=='24146c612379d086bd573185160a72ee17c1b6593ef499153360118cb4aed13a'
prior=json.loads((root/'experiments/bp7_full_cos200_final/e180_stage_last/submission_validation.json').read_text())
assert prior['mean_gpu_replay_all_members_byte_identical']
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.03)
original_load=torch.load
def load_on_gpu(path,*args,**kwargs):
    assert Path(path)==raw
    kwargs['map_location']='cuda:0'
    return original_load(path,*args,**kwargs)
predict.torch.load=load_on_gpu
images=root.parent/'cs701_data/cs701-sar-course-data/val'
command=['predict.py','--run',str(suite/'full_train_e200'),'--from-raw',str(raw),'--images',str(images),'--images-json',str(images/'images.json'),'--shift','-0.2','--cls-lambda','2','--stage-box','mean','--soft-sigma','0.5','--out',str(out)]
write_json(work/'launch.json',dict(pid=os.getpid(),stage='postprocessing',command=command,raw=str(raw),memory_fraction=0.03,automatic_upload=False))
sys.argv=command
predict.main()
with zipfile.ZipFile(base) as a,zipfile.ZipFile(out) as b:
    assert b.testzip() is None and set(b.namelist())=={'classification.csv','detection.json'}
    assert a.read('classification.csv')==b.read('classification.csv')
    detection_changed=a.read('detection.json')!=b.read('detection.json')
result=validate_zip(out,images/'images.json')
result.update(epoch=180,classification_lambda=2.0,stage_box='mean',soft_sigma=0.5,source_submission=str(base),source_submission_sha256=sha256_file(base),source_checkpoint_sha256=prior['source_checkpoint_sha256'],raw_cache=str(raw),raw_sha256=prior['raw_sha256'],classification_bytes_identical=True,detection_changed=detection_changed,baseline_gpu_reproduction_validation=str(root/'experiments/bp7_full_cos200_final/e180_stage_last/submission_validation.json'),allocated_peak_bytes=torch.cuda.max_memory_allocated(),crc_passed=True,validated_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),official_score=None,automatic_upload=False,change='Soft-NMS sigma0.75 ->0.5 only; same e180 raw forward, mean boxes, lambda2, shift-0.2 and remaining frozen defaults.')
write_json(work/'submission_validation.json',result)
with out.with_suffix('.sha256').open('x') as f: f.write(result['sha256']+'  '+out.name+'\n')
launch=json.loads((work/'launch.json').read_text()); launch.update(stage='complete_validated',validation=str(work/'submission_validation.json')); write_json(work/'launch.json',launch)
print(json.dumps(result,indent=2),flush=True)
