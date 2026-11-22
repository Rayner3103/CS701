"""Frozen DINOv3 ViT-B/16 embeddings for train / official val / official test, and the nearest-train cosine of every
image, plain and max over the 8 flip/rot90 (D4) transforms of the query. Output: nn.npz"""
import sys, json, numpy as np, torch, timm
from pathlib import Path
from PIL import Image

ROOT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('data')  # course data root with train/ and val/
TEST = Path(sys.argv[1])          # extracted official test folder
OUT = Path(sys.argv[2])
dev = 'cuda'
m = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0).to(dev).eval().half()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
S = 256


def files(split_dir, meta):
    ims = json.loads((split_dir / meta).read_text())
    ims = ims['images'] if isinstance(ims, dict) else ims
    return [im['id'] for im in ims], [split_dir / im['file_name'] for im in ims]


def load(p):
    a = np.asarray(Image.open(p).convert('L').resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.
    return torch.from_numpy(a)


@torch.no_grad()
def embed(paths, d4=False):
    out = []
    for i in range(0, len(paths), 64):
        x = torch.stack([load(p) for p in paths[i:i + 64]]).to(dev)[:, None].repeat(1, 3, 1, 1)
        views = [x]
        if d4:
            views = [torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)]
        f = [torch.nn.functional.normalize(m(((v - mean) / std).half()).float(), dim=1) for v in views]
        out.append(torch.stack(f, 1).cpu())          # B x V x D
    return torch.cat(out)


def train_list():
    rows = [l.split(',') for l in (ROOT / 'train/labels.csv').read_text().splitlines()[1:]]
    return [int(r[0]) for r in rows], [ROOT / 'train' / r[1] for r in rows]


tid, tp = train_list()
T = embed(tp)[:, 0].to(dev)                      # N x D
res = {'train_id': np.array(tid)}
# train leave-one-out, plain
sim = T @ T.T
sim.fill_diagonal_(-1)
v, j = sim.max(1)
res['train_loo'] = v.cpu().numpy(); res['train_loo_j'] = j.cpu().numpy()
for name, d, meta in (('val', ROOT / 'val', 'images.json'), ('test', TEST, 'images.json')):
    ids, ps = files(d, meta)
    Q = embed(ps, d4=True).to(dev)               # M x 8 x D
    s = torch.einsum('mvd,nd->mvn', Q, T)        # M x 8 x N
    plain_v, plain_j = s[:, 0].max(1)
    d4_v, d4_flat = s.flatten(1).max(1)
    res[f'{name}_id'] = np.array(ids)
    res[f'{name}_plain'] = plain_v.cpu().numpy(); res[f'{name}_plain_j'] = plain_j.cpu().numpy()
    res[f'{name}_d4'] = d4_v.cpu().numpy(); res[f'{name}_d4_j'] = (d4_flat % T.shape[0]).cpu().numpy()
    res[f'{name}_d4_view'] = (d4_flat // T.shape[0]).cpu().numpy()
    res[f'{name}_path'] = np.array([str(p) for p in ps])
res['train_path'] = np.array([str(p) for p in tp])
np.savez(OUT, **res)
for k in ('train_loo', 'val_plain', 'val_d4', 'test_plain', 'test_d4'):
    a = res[k]
    print(k, 'median %.3f' % np.median(a), ' '.join('>=%.2f:%.1f%%' % (t, 100 * (a >= t).mean()) for t in (.99, .97, .95, .94, .92, .90)))
