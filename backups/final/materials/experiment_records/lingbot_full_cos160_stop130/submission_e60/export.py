from pathlib import Path
from zoneinfo import ZoneInfo
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback

ROOT = Path('/mnt/hdd/junkaichen/CS701/CS701')
WORK = ROOT / 'experiments/lingbot_full_cos160_stop130/submission_e60'
SUITE = Path('/mnt/hdd/junkaichen/CS701/cs701_runs/LingBotB_full9392_cos160_stop130_s0_20261031_nocompile')
CODE, RUN = SUITE / 'code', SUITE / 'full_train'
OFFICIAL = Path('/mnt/hdd/junkaichen/CS701/cs701_data/cs701-sar-course-data/val')
OUT = ROOT / 'submissions/lingb_e60.zip'
PYTHON = '/home/junkaichen@smu.edu.sg/miniconda3/envs/cs701bench/bin/python'


def now():
    return datetime.datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(name, value):
    path = WORK / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


state = dict(pid=os.getpid(), started_at=now(), epoch=60, suite=str(SUITE),
             output=str(OUT), gpu=1, batch_size=1, allocator_fraction=0.035, uploaded=False)


def update(stage, **extra):
    state.update(stage=stage, updated_at=now(), **extra)
    save('status.json', state)
    print(now(), stage, extra, flush=True)


os.sched_setaffinity(0, {0, 1, 2, 3})
os.nice(10)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
           PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
           HF_HOME='/mnt/hdd/junkaichen/hf-cache', HF_HUB_OFFLINE='1')
for name in ['PYTORCH_ALLOC_CONF', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE']:
    env.pop(name, None)

try:
    update('verifying_inputs')
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite output')
    verified = 0
    for line in (SUITE / 'source.sha256').read_text().splitlines():
        digest, relative = line.split(None, 1)
        if sha(CODE / relative) != digest:
            raise ValueError('Frozen source hash changed: ' + relative)
        verified += 1
    checkpoint = RUN / 'ckpt_e60.pt'
    checksum = sha(checkpoint)
    saved = json.loads((RUN / 'log.txt').read_text().splitlines()[0])
    assert saved['backbone'] == 'lingbot_b' and saved['train_split'] == 'train+val+test'
    images = json.loads((OFFICIAL / 'images.json').read_text())['images']
    assert len(images) == 1426
    save('smoke_images.json', dict(images=images[:8]))
    sys.path[:0] = [str(CODE), str(CODE / 'tools')]
    from run_bp7_full_stop130 import validate_zip
    save('provenance.json', dict(
        created_at=now(), suite=str(SUITE), checkpoint=str(checkpoint), checkpoint_sha256=checksum,
        epoch=60, training_images=9392, schedule_epochs=160, source_files_verified=verified,
        prediction_images=1426, images_json_sha256=sha(OFFICIAL / 'images.json'), batch_size=1,
        shift=-0.2, gpu=1, allocator_fraction=0.035, uploaded=False,
        inference='single image once, frozen predict.py FINAL defaults; no TTA/ensemble'))
    for phase, image_list, destination in [
        ('smoke', WORK / 'smoke_images.json', WORK / 'smoke.zip'),
        ('full', OFFICIAL / 'images.json', WORK / 'submission.pending.zip'),
    ]:
        command = [PYTHON, '-u', str(WORK / 'limited_predict.py'), str(CODE), '--run', str(RUN),
                   '--weights', 'ckpt_e60.pt', '--images', str(OFFICIAL), '--images-json', str(image_list),
                   '--batch-size', '1', '--workers', '0', '--shift', '-0.2', '--out', str(destination)]
        save(phase + '_command.json', command)
        with (WORK / (phase + '_predict.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
            update(phase + '_prediction', child_pid=child.pid)
            while True:
                try:
                    result_code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    update(phase + '_prediction')
        if result_code:
            raise subprocess.CalledProcessError(result_code, command)
        result = validate_zip(destination, image_list)
        save(phase + '_validation.json', result)
    if sha(checkpoint) != checksum:
        raise ValueError('Checkpoint changed during inference')
    # Exclusive publication preserves any pre-existing submission.
    os.link(WORK / 'submission.pending.zip', OUT)
    result = validate_zip(OUT, OFFICIAL / 'images.json')
    result.update(epoch=60, checkpoint=str(checkpoint), checkpoint_sha256=checksum,
                  backbone='lingbot_b', finished_at=now())
    save('submission_validation.json', result)
    update('completed', child_pid=None, submission=result)
except BaseException as error:
    update('failed', error=type(error).__name__ + ': ' + str(error))
    traceback.print_exc()
    raise
