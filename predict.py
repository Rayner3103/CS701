"""Predict on the released (unlabelled) val or test images with a trained run and write submission.zip for Codabench.
Example: python predict.py --run runs/full_seed0/terramind_pretrained_moelora_sp --images ../dataset-v2/val"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from make_submission import write_submission
from sarbench.adapters import adapter_kwargs, add_adapters
from sarbench.backbones import build_backbone
from sarbench.data import MEAN, STD
from sarbench.model import MultiTaskModel


class Unlabelled(Dataset):
    """The images of images.json, preprocessed as in sarbench/data.py without flips. An item is (image, file_name)."""

    def __init__(self, root, images, size=512):
        self.root, self.images = Path(root), images
        self.transform = v2.Compose([v2.ToImage(), v2.Resize((size, size)),
                                     v2.ToDtype(torch.float32, scale=True), v2.Normalize([MEAN], [STD])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        name = self.images[i]['file_name']
        return self.transform(Image.open(self.root / name).convert('L')), name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='a run folder of train.py (has metrics.json)')
    p.add_argument('--weights', default='best_model.pt', help='best_model.pt or model.pt (last epoch)')
    p.add_argument('--images', type=Path, required=True, help='the split folder holding images.json')
    p.add_argument('--out', type=Path, default=Path('submission.zip'))
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    # The model as the run built it; pretrained=False: every weight comes from the checkpoint
    run = json.loads((args.run / 'metrics.json').read_text())['args']
    backbone = build_backbone(run['backbone'], pretrained=False)
    if run['adapt'] != 'full':
        add_adapters(backbone, run['adapt'], **adapter_kwargs(run))
    model = MultiTaskModel(backbone, task_routing=run['adapt'] == 'moelora',
                           spatial_prior=run.get('spatial_prior', False))
    model.load_state_dict(torch.load(args.run / args.weights, map_location='cpu'))
    model.cuda().eval()

    images = json.loads((args.images / 'images.json').read_text())['images']
    loader = DataLoader(Unlabelled(args.images, images), args.batch_size, num_workers=args.workers)
    preds = {}
    with torch.no_grad():
        for batch, names in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, outputs, _ = model(batch.cuda())
            for name, lg, out in zip(names, logits, outputs):
                preds[name] = (lg.argmax().item(), out['boxes'].float().tolist(),
                               (out['labels'] - 1).tolist(),  # detector labels 1..9; write_submission adds 1
                               out['scores'].float().tolist())
            print(f'\r{len(preds)}/{len(images)}', end='', flush=True)

    write_submission(images, preds, out=args.out)
    n_boxes = sum(len(v[1]) for v in preds.values())
    print(f'\nwrote {args.out}: {len(preds)} images, {n_boxes} boxes')


if __name__ == '__main__':
    main()
