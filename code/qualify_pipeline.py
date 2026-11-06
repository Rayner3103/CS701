"""Exercise full F2-style validation and native submission export on nine TRAIN images only."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
import train
from run_project import evaluate_native
from run_safety import atomic_json
from sarbench.data import SARMultiTask, collate


def main():
    torch.set_num_threads(4)
    root=Path('D:/CS701_junkai_ib')
    payload=torch.load(root/'qualification_checkpoint.pt',map_location='cpu',weights_only=False)
    config=payload.get('config',payload['args']);args=SimpleNamespace(**config);args.init='scratch'
    model=train.build_model(args);model.load_state_dict(payload['model'],strict=True);model.cuda().eval()
    dataset=SARMultiTask(args.data,'train',native_size=True,mean=args.mean,std=args.std)
    indices=[next(i for i,row in enumerate(dataset.rows) if int(row['label'])==label) for label in range(9)]
    ids={int(dataset.coco_images[dataset.rows[i]['image']]['id']) for i in indices}
    coco=json.loads((Path(args.data)/'detection/instances_train.json').read_text(encoding='utf-8'))
    small={**coco,'images':[i for i in coco['images'] if int(i['id']) in ids],
           'annotations':[a for a in coco['annotations'] if int(a['image_id']) in ids]}
    gt=root/'pipeline_check_gt.json';atomic_json(gt,small)
    loader=DataLoader(Subset(dataset,indices),batch_size=4,shuffle=False,num_workers=0,collate_fn=collate)
    metrics,predictions=evaluate_native(model,loader,gt,torch.device('cuda'))
    assert 0<=metrics['macro_f1']<=1 and 0<=metrics['mAP']<=1
    inputs=root/'pipeline_check_inputs';inputs.mkdir(exist_ok=True)
    atomic_json(inputs/'images.json',{'images':small['images']})
    # Upgrade this bounded qualification artifact to the standalone full-checkpoint inference interface.
    payload['config']=config;payload['completed_epoch']=0
    torch.save(payload,root/'qualification_checkpoint.pt')
    report={'status':'PASSED','scope':'nine train-only images; validation/COCO/postprocessing execution check, not accuracy evidence',
            'images':9,'predictions':len(predictions),'metric_fields_finite':True,
            'postprocessing':'F2 soft-NMS/objectness/stage mean/class fusion, coordinate shift0',
            'model_epoch':0}
    atomic_json(root/'pipeline_qualification.json',report);print(json.dumps(report))


if __name__=='__main__':main()
