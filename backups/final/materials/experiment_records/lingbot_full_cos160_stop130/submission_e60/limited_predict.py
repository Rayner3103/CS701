import json
import os
import runpy
import sys
from pathlib import Path

code = Path(sys.argv[1])
arguments = sys.argv[2:]
sys.path.insert(0, str(code))
os.chdir(code)

import torch
from sarbench.data import SARImages

torch.set_num_threads(1)
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.035, device=0)
original_getitem = SARImages.__getitem__


def progress_getitem(self, index):
    item = original_getitem(self, index)
    if index % 100 == 0:
        print(f'PREDICTION_PROGRESS image_index={index}', flush=True)
    return item


SARImages.__getitem__ = progress_getitem
sys.argv = [str(code / 'predict.py'), *arguments]
print('GPU_RESOURCE_LIMIT ' + json.dumps(dict(
    visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'), allocator_fraction=0.035)), flush=True)
try:
    runpy.run_path(str(code / 'predict.py'), run_name='__main__')
finally:
    print('GPU_RESOURCE_PEAK ' + json.dumps(dict(
        allocated_bytes=torch.cuda.max_memory_allocated(0),
        reserved_bytes=torch.cuda.max_memory_reserved(0))), flush=True)
