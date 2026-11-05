import os,sys,json
from pathlib import Path
root=Path('/mnt/hdd/junkaichen/CS701/CS701')
suite=root.parent/'cs701_runs/BP7_full9392_cos200_e160to200_s0_20261101_v2'
work=root/'experiments/bp7_full_cos200_final/e180_stage_last'
sys.path[:0]=[str(suite/'code'),str(suite/'code/tools')]
import torch
import predict
from run_bp7_schedule_search import write_json
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
if os.getpriority(os.PRIO_PROCESS,0)<10: os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.03)
raw=root/'experiments/bp7_full_cos200_final/e180_immediate/raw.pt'
original_load=torch.load
def load_on_gpu(path,*args,**kwargs):
    assert Path(path)==raw
    kwargs['map_location']='cuda:0'
    return original_load(path,*args,**kwargs)
predict.torch.load=load_on_gpu
images=root.parent/'cs701_data/cs701-sar-course-data/val'
outputs={'mean':work/'mean_gpu_reproduction.zip','last':root/'submissions/BP7_e180_last_cls2.zip'}
for stage,out in outputs.items():
    assert not out.exists()
    sys.argv=['predict.py','--run',str(suite/'full_train_e200'),'--from-raw',str(raw),'--images',str(images),'--images-json',str(images/'images.json'),'--shift','-0.2','--cls-lambda','2','--stage-box',stage,'--out',str(out)]
    predict.main()
    torch.cuda.empty_cache()
write_json(work/'gpu_replay_complete.json',dict(outputs={k:str(v) for k,v in outputs.items()},allocated_peak_bytes=torch.cuda.max_memory_allocated(),automatic_upload=False))
