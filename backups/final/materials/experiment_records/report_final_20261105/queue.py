#!/usr/bin/env python3
"""Report experiments: seed replicates of the final recipe and leave-one-out ablations (split trainvl 8,400 / vallike 992).
Two GPU lanes; a failed training resumes from training_state.pt up to 3 times. Delete a job from jobs.json to skip it.
Usage: queue.py [--smoke]"""
import json, os, subprocess, sys, threading, time
from pathlib import Path
E = Path(__file__).resolve().parent; RUNS = Path('/mnt/hdd/junkaichen/CS701/cs701_runs')
PY = '/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python'
DATA = '/mnt/hdd/junkaichen/CS701/cs701_data/dataset/SARFact-Course-20K'
SMOKE = '--smoke' in sys.argv
ANCH = ['--anchor-offset', '0.5', '--anchor-sizes=6,8,11;16,19,23;32,38,45;64,76,91;128,152,181']
def recipe(rank=80, alpha=160, rep=1536, sp=64, norms=True, cascade=True, sizeiou=True, nwd=8, flip=-0.3, anchors=True):
    a = ['--backbone', 'dinov3', '--init', 'pretrained', '--adapt', 'lora', '--lora-targets', 'attn.qkv,attn.proj,mlp.fc1,mlp.fc2',
         '--lora-rank', str(rank), '--lora-alpha', str(alpha), '--backbone-lr', '0.0003', '--lr', '0.0001', '--weight-decay', '0.05',
         '--aligned-roi', '--grad-ckpt', '--warmup-epochs', '1', '--reproducible-loader', '--batch-size', '16', '--workers', '4', '--eval-batch-size', '4']
    if sp: a += ['--spatial-prior', str(sp)]
    if norms: a += ['--train-norms']
    if cascade: a += ['--cascade', '--cascade-rep', str(rep)]
    if cascade and sizeiou: a += ['--cascade-size-iou']
    if nwd: a += ['--rpn-nwd', str(nwd)]
    if flip: a += ['--flip-bias', str(flip)]
    if anchors: a += ANCH
    return a
VARIANTS = {
    'base': dict(), 'noSP': dict(sp=0), 'noSizeIoU': dict(sizeiou=False), 'noFlipBias': dict(flip=0), 'noNWD': dict(nwd=0),
    'noCascade': dict(cascade=False), 'noSmallAnchors': dict(anchors=False), 'noTrainNorms': dict(norms=False),
    'smallCapacity': dict(rank=32, alpha=64, rep=1024),
}
def loo(v, seed):
    return dict(name=f'RPT_LOO36_{v}_s{seed}', kind='loo', seed=seed, shift=-0.4 if v == 'noFlipBias' else -0.2,
                flags=recipe(**VARIANTS[v]) + ['--epochs', '36', '--schedule-epochs', '36', '--eval-every', '36', '--save-training-state'])
def twin(seed):
    return dict(name=f'RPT_TWIN_cos200_s{seed}', kind='twin', seed=seed, shift=-0.2,
                flags=recipe() + ['--epochs', '200', '--schedule-epochs', '200', '--stop-after-epoch', '180', '--save-every', '10', '--eval-every', '20', '--save-training-state'])
REF = dict(name='RPT_REF24_s0', kind='ref', seed=0, shift=None,
           flags=['--backbone', 'vit', '--init', 'pretrained', '--adapt', 'lora', '--batch-size', '16', '--workers', '4', '--eval-batch-size', '4', '--epochs', '24', '--eval-every', '24'])
LANE0 = [twin(1), twin(2)]
SHARED = [loo(v, 0) for v in VARIANTS] + [REF] + [loo(v, 1) for v in VARIANTS] + [loo(v, 2) for v in VARIANTS]
if SMOKE:
    LANE0 = []; SHARED = [loo(v, 0) for v in VARIANTS] + [twin(1), REF]
lock = threading.Lock()
def log(m):
    with lock, open(E / 'logs' / ('smoke.log' if SMOKE else 'queue.log'), 'a') as f:
        f.write(f"{time.strftime('%F %T')} {m}\n")
def skip(name):
    p = E / 'skip.txt'
    return p.exists() and name in p.read_text().split()
def train(job, gpu):
    out = RUNS / (job['name'] + ('_smoke' if SMOKE else '')); rd = out / 'run'; out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, 'train.py', '--data', DATA, '--train-split', 'trainvl', '--eval-splits', 'vallike', '--seed', str(job['seed'])] + job['flags'] + ['--out', str(out), '--name', 'run']
    if SMOKE:
        cmd = [PY, 'train.py', '--data', DATA, '--train-split', 'trainvl', '--eval-splits', 'vallike', '--seed', '0'] + [x for x in job['flags']] + ['--out', str(out), '--name', 'run', '--limit', '64', '--epochs', '1', '--eval-every', '1']
        cmd = [c for c in cmd]
        for i, c in enumerate(cmd):
            if c == '--schedule-epochs': cmd[i + 1] = '1'
            if c == '--stop-after-epoch': cmd[i + 1] = '1'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TMPDIR='/mnt/hdd/junkaichen/CS701/CS701/.runtime/tmp')
    for attempt in range(4):
        if (rd / 'model.pt').exists() and job['kind'] != 'twin' or (job['kind'] == 'twin' and (rd / f'ckpt_e180.pt').exists() and not SMOKE):
            return True
        c = list(cmd)
        if (rd / 'training_state.pt').exists(): c += ['--resume', str(rd / 'training_state.pt')]
        log(f"gpu{gpu} train {job['name']} attempt {attempt}")
        with open(out / 'stdout.log', 'a') as f:
            rc = subprocess.call(['nice', '-n', '10'] + c, cwd=E / 'code', env=env, stdout=f, stderr=subprocess.STDOUT)
        log(f"gpu{gpu} train {job['name']} rc={rc}")
        if rc == 0: return True
        time.sleep(300)
    return False
PRED = ['--split', 'vallike', '--batch-size', '4', '--workers', '2', '--rpn-topn', '2000', '--obj-pow', '0', '--soft-sigma', '0.5', '--cls-lambda', '2', '--fusion', 'agn', '--gamma', '1', '--topk', '9', '--dump-raw']
def predict(job, gpu, weights, tag):
    rd = RUNS / (job['name'] + ('_smoke' if SMOKE else '')) / 'run'; ev = E / 'raws' / job['name']; ev.mkdir(parents=True, exist_ok=True)
    dst = ev / f'{tag}_vallike.pt'
    if dst.exists(): return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2')
    with open(ev / f'{tag}.predict.log', 'w') as f:
        rc = subprocess.call(['nice', '-n', '10', PY, 'predict.py', '--run', str(rd), '--weights', str(weights), '--shift', str(job['shift'])] + PRED, cwd=E / 'code', env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"gpu{gpu} predict {job['name']} {tag} rc={rc}")
    if rc == 0: os.replace(rd / 'raw_vallike.pt', dst)
def post(job, gpu):
    rd = RUNS / (job['name'] + ('_smoke' if SMOKE else '')) / 'run'
    if job['kind'] == 'ref':
        return
    if job['kind'] == 'loo' or SMOKE:
        predict(job, gpu, 'model.pt', 'final'); (rd / 'training_state.pt').unlink(missing_ok=True); return
    avg = E / 'raws' / job['name'] / 'avg100-180s10.pt'; avg.parent.mkdir(parents=True, exist_ok=True)
    if not avg.exists():
        subprocess.call([PY, str(E / 'eval_tools' / 'average_local.py'), '--dir', str(rd), '--epochs'] + [str(e) for e in range(100, 181, 10)] + ['--out', str(avg)], cwd=E / 'code')
    predict(job, gpu, avg, 'avg100-180s10')
    for e in (130, 160, 180): predict(job, gpu, rd / f'ckpt_e{e}.pt', f'e{e}')
def worker(gpu, own):
    while True:
        with lock:
            if own: job = own.pop(0)
            elif SHARED: job = SHARED.pop(0)
            else: return
        if skip(job['name']): log(f"skip {job['name']}"); continue
        t = time.time()
        try:
            ok = train(job, gpu)
            if ok: post(job, gpu)
            log(f"gpu{gpu} done {job['name']} ok={ok} {(time.time()-t)/3600:.2f}h")
        except Exception as ex:
            log(f"gpu{gpu} ERROR {job['name']} {ex!r}")
log(f"start smoke={SMOKE}")
ts = [threading.Thread(target=worker, args=(0, LANE0))] + ([threading.Thread(target=worker, args=(1, []))] )
[t.start() for t in ts]; [t.join() for t in ts]
log('ALL DONE')
