"""Infer from this project's full checkpoint and write a local upload-ready ZIP.

Does not upload to a competition. Use this entry instead of the unchanged upstream
predict.py, which expects a different checkpoint format and square-resize inputs.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zipfile

import torch
from torch.utils.data import DataLoader

import train
from native_postprocessing import FINAL_POST, fused_class
from run_safety import atomic_json, sha256
from sarbench.data import SARImages, to_original_xywh


def collate_images(batch):
    images,metadata=zip(*batch)
    return torch.stack(images),list(metadata)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    opts=parser.parse_args()
    opts.out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(opts.checkpoint,map_location='cpu',weights_only=False)
    saved=dict(payload['config']);saved['init']='scratch'
    model=train.build_model(SimpleNamespace(**saved));model.load_state_dict(payload['model'],strict=True)
    model.cuda().eval();model.roi_heads.post.update(FINAL_POST)
    info=json.loads((opts.images/'images.json').read_text(encoding='utf-8'))
    images=info['images'] if isinstance(info,dict) else info
    assert len({int(i['id']) for i in images})==len(images)
    dataset=SARImages(opts.images,images,native_size=True,mean=saved['mean'],std=saved['std'])
    loader=DataLoader(dataset,batch_size=opts.batch_size,shuffle=False,num_workers=0,collate_fn=collate_images)
    predictions=[];seen=[]
    with (opts.out/'classification.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['image_id','label'])
        with torch.inference_mode():
            for batch,metadata in loader:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,outputs,_=model(batch.cuda(),image_sizes=[m['valid_size'] for m in metadata],return_raw=True)
                for lg,output,meta in zip(logits,outputs,metadata):
                    image_id=int(meta['image_id']);seen.append(image_id);writer.writerow([image_id,fused_class(lg,output)])
                    boxes=to_original_xywh(output['boxes'],meta['orig_size'],valid_size=meta['valid_size'],
                                           scale_xy=meta['scale_xy']).tolist()
                    assert len(boxes)<=100
                    for box,label,score in zip(boxes,output['labels'].tolist(),output['scores'].tolist()):
                        if not (all(math.isfinite(v) for v in box) and math.isfinite(score)):
                            raise ValueError('Nonfinite prediction')
                        if box[2]>0 and box[3]>0:
                            assert 1<=label<=9 and 0<=score<=1
                            predictions.append({'image_id':image_id,'category_id':label,'bbox':box,'score':score})
    assert seen==[int(i['id']) for i in images]
    (opts.out/'detection.json').write_text(json.dumps(predictions),encoding='utf-8')
    with zipfile.ZipFile(opts.out/'submission.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('classification.csv','detection.json'):archive.write(opts.out/name,name)
    atomic_json(opts.out/'INFERENCE_PROVENANCE.json',{'checkpoint':str(opts.checkpoint),
                'checkpoint_sha256':sha256(opts.checkpoint),'source_epoch':payload['completed_epoch'],
                'postprocessing':FINAL_POST,'images':len(images),'detections':len(predictions),'uploaded':False})


if __name__=='__main__':main()
