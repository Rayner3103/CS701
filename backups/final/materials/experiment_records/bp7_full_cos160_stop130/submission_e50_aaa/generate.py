"""Generate the requested aaa.zip from the already saved full-data BP7 e50."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys

META = Path(__file__).resolve().parent
ROOT = META.parents[2]
SUITE = ROOT.parent / 'cs701_runs/BP7_full9392_cos160_stop130_s0_20261031'
CODE = SUITE / 'code'
RUN = SUITE / 'full_train'
OFFICIAL = ROOT.parent / 'cs701_data/cs701-sar-course-data/val'
OUT = ROOT / 'submissions/aaa.zip'
TEMP = META / 'aaa.pending.zip'
for directory in (CODE, CODE / 'tools'):
    sys.path.insert(0, str(directory))
from run_bp7_schedule_search import stamp, write_json, sha256_file, verify_snapshot, read_json
from run_bp7_full_stop130 import validate_zip

lock = (META / 'driver.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if OUT.exists() or (META / 'status.json').exists():
    raise RuntimeError('Refusing to overwrite an existing submission or execution')
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
os.environ.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                  TOKENIZERS_PARALLELISM='false', HF_HOME='/mnt/hdd/junkaichen/hf-cache', HF_HUB_OFFLINE='1')
state = dict(pid=os.getpid(), started_at=stamp(), checkpoint_epoch=50, output=str(OUT), uploaded=False)
def update(stage, **extra):
    state.update(stage=stage, updated_at=stamp(), **extra)
    write_json(META / 'status.json', state)
    print(stamp(), stage, extra, flush=True)

try:
    update('verifying_inputs')
    manifest = read_json(SUITE / 'launch_manifest.json')
    verify_snapshot(SUITE, Path(manifest['dependency_suite']), Path(manifest['data']))
    saved = json.loads((RUN / 'log.txt').read_text().splitlines()[0])
    assert saved['train_split'] == 'train+val+test' and saved['schedule_epochs'] == 160
    assert saved['lr'] == 1e-4 and saved['backbone_lr'] == 3e-4 and saved['roi_context'] == 0
    checkpoint = RUN / 'ckpt_e50.pt'
    digest = sha256_file(checkpoint)
    prior = ROOT.parent / 'cs701_runs/BP7_LR130_local_s0_20261029_v2/lr1x/submission/launch.json'
    wrapper = read_json(prior)['argv'][2]
    command = [sys.executable, '-c', wrapper, '--run', str(RUN), '--weights', checkpoint.name,
               '--images', str(OFFICIAL), '--batch-size', '2', '--workers', '0',
               '--shift', '-0.2', '--out', str(TEMP)]
    from pp_eval import FINAL, LAM
    post = dict(FINAL, shift=-0.2, cls_lambda=LAM)
    write_json(META / 'launch.json', dict(started_at=stamp(), argv=command, cwd=str(CODE),
               checkpoint=str(checkpoint), checkpoint_sha256=digest, checkpoint_epoch=50,
               schedule_epochs=160, training_images=9392, official_prediction_images=1426,
               gpu=1, gpu_allocator_fraction_limit=0.12, batch_size=2, workers=0, post=post,
               images_json_sha256=sha256_file(OFFICIAL / 'images.json'), output=str(OUT), automatic_upload=False))
    with (META / 'predict.log').open('w') as log:
        with subprocess.Popen(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT) as child:
            update('predicting', child_pid=child.pid)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    update('predicting', child_pid=child.pid)
    if code:
        raise subprocess.CalledProcessError(code, command)
    update('validating')
    validation = validate_zip(TEMP, OFFICIAL / 'images.json')
    assert validation['prediction_images'] == 1426
    assert sha256_file(checkpoint) == digest
    if OUT.exists():
        raise RuntimeError('Output appeared during prediction; refusing overwrite')
    TEMP.rename(OUT)
    validation.update(path=str(OUT), time=stamp(), checkpoint=str(checkpoint), checkpoint_sha256=digest,
                      checkpoint_epoch=50, schedule_epochs=160, training_images=9392, post=post)
    write_json(META / 'validation.json', validation)
    OUT.with_suffix('.sha256').write_text(validation['sha256'] + '  ' + OUT.name + '\n')
    update('complete', submission=str(OUT), bytes=validation['bytes'], boxes=validation['boxes'])
except Exception as error:
    update('failed', error=f'{type(error).__name__}: {error}')
    raise
