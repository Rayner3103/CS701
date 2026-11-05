"""Equal parameter average of the requested cos200 snapshots, then one-model inference."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path('/mnt/hdd/junkaichen/CS701/CS701')
RUNS = ROOT.parent / 'cs701_runs'
WORK = Path(__file__).resolve().parent
RESEARCH = WORK.parent
SUITE = RUNS / 'BP7_full9392_cos200_e160to200_s0_20261101_v2'
PREVIOUS = RESEARCH / 'checkpoint_average_e160_e170_e180'
OUTPUT = ROOT / 'submissions/BP7_avg100120140160_rpn2000_obj0_soft05_cls2.zip'
MODEL = WORK / 'averaged_model.pt'
SOURCES = [
    (100, RUNS / 'BP7_full9392_cos200_stop160_s0_20261031_expandable/full_train/ckpt_e100.pt', '623c2622fad60911a257592d374a2f26cb24ee5c56df36a623a1ff743f54b48f'),
    (120, RUNS / 'BP7_full9392_cos200_stop160_s0_20261031_expandable/full_train/ckpt_e120.pt', '8a18baad3e1a2034b9e0d4d76a8a5e420f4586356fefbc45c09d853120d3dde3'),
    (140, RUNS / 'BP7_full9392_cos200_e130to160_s0_20261101_mem35/full_train_e160/ckpt_e140.pt', '2a0b51bb9ff9da2d398ef799a5e378bbd9518e267901ae4c88aea2659b25a1e3'),
    (160, RUNS / 'BP7_full9392_cos200_e147to160_s0_20261101_tmp/full_train_e160/ckpt_e160.pt', 'a8de35f3209f2772b5f5c262c3b79a262a568ba570feb623e449157d98ded0de'),
]


def stamp():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    if os.nice(0) < 10:
        os.nice(10 - os.nice(0))
    import torch
    torch.set_num_threads(1)
    assert not MODEL.exists() and not OUTPUT.exists(), 'Preserve existing artifacts'
    states = []
    provenance = []
    for epoch, path, expected in SOURCES:
        actual = sha(path)
        assert actual == expected, (epoch, actual)
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert len(state) == 234
        assert sum(t.numel() for t in state.values()) == 39867360
        assert all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in state.values())
        if states:
            assert state.keys() == states[0].keys()
            assert all(t.shape == states[0][k].shape for k, t in state.items())
        states.append(state)
        provenance.append(dict(epoch=epoch, path=str(path), sha256=actual, coefficient=.25))
    averaged = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float64)
        for state in states:
            acc.add_(state[key].double())
        averaged[key] = acc.div_(4).to(torch.float32)
        assert torch.isfinite(averaged[key]).all().item()
    with MODEL.open('xb') as f:
        torch.save(averaged, f)
    reloaded = torch.load(MODEL, map_location='cpu', weights_only=True)
    assert all(torch.equal(t, reloaded[k]) for k, t in averaged.items())
    model_sha = sha(MODEL)
    avg_meta = read(PREVIOUS / 'average_validation.json')
    avg_meta.update(created_at=stamp(), method='Elementwise equal parameter average of e100/e120/e140/e160, accumulated in float64 and stored float32', sources=provenance, output=str(MODEL), output_sha256=model_sha, official_submission_ready=False)
    avg_meta.pop('full_prediction_validation', None)
    avg_meta['trajectory_audit'] = 'Independent CPU audit: same cos200 trajectory; source manifests and e130/e147 continuation hashes checked; core train/predict/model code hashes identical.'
    write(WORK / 'average_validation.json', avg_meta)
    print('PARAMETER_AVERAGE_READY', model_sha, flush=True)
    del states, state, averaged, reloaded

    launch = read(PREVIOUS / 'launch.json')
    command = launch['command']
    for flag, value in {'--weights': str(MODEL), '--out': str(OUTPUT), '--raw-out': str(WORK / 'raw.pt')}.items():
        command[command.index(flag) + 1] = value
    env = os.environ.copy()
    for key in ('PYTORCH_ALLOC_CONF', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    temp = ROOT / '.runtime/avg100120140160'
    temp.mkdir(parents=True, exist_ok=True, mode=0o700)
    env.update(CUDA_VISIBLE_DEVICES='1', HF_HOME='/mnt/hdd/junkaichen/hf-cache', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp))
    with (WORK / 'predict.log').open('x') as log:
        proc = subprocess.Popen(command, cwd=SUITE / 'code', env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch.update(started_at=stamp(), pid=proc.pid, command=command, stage='running', source_epochs=[100, 120, 140, 160], coefficients=[.25]*4, checkpoint=str(MODEL), checkpoint_sha256=model_sha, output=str(OUTPUT))
        launch.pop('submission_validation', None)
        write(WORK / 'launch.json', launch)
        code = proc.wait()
    launch.update(returncode=code, finished_at=stamp(), stage='prediction_complete' if code == 0 else 'prediction_failed')
    write(WORK / 'launch.json', launch)
    assert code == 0, f'Prediction failed: {code}; see predict.log'

    sys.path[:0] = [str(SUITE / 'code'), str(SUITE / 'code/tools')]
    from run_bp7_cos200_final import validate_zip
    manifest = ROOT.parent / 'cs701_data/cs701-sar-course-data/val/images.json'
    validation = validate_zip(OUTPUT, manifest)
    baseline = read(PREVIOUS / 'submission_validation.json')
    def classifications(path):
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            assert set(z.namelist()) == {'classification.csv', 'detection.json'}
            return list(csv.reader(io.StringIO(z.read('classification.csv').decode())))
    rows, oldrows = classifications(OUTPUT), classifications(baseline['path'])
    assert len(rows) == len(oldrows) == 1427 and rows[0] == oldrows[0]
    assert [r[0] for r in rows] == [r[0] for r in oldrows]
    changed = sum(a != b for a, b in zip(rows[1:], oldrows[1:]))
    assert sha(MODEL) == model_sha
    validation.update(method='Equal parameter average then single-model inference', source_epochs=[100,120,140,160], coefficients=[.25]*4, sources=provenance, checkpoint=str(MODEL), checkpoint_sha256=model_sha, rpn_test_topn=2000, classification_lambda=2, stage_box='mean', soft_sigma=.5, obj_pow=0, raw_cache=str(WORK/'raw.pt'), raw_sha256=sha(WORK/'raw.pt'), changed_classification_rows_vs_avg160170180=changed, parameter_count=125076960, trainable_parameter_count=39867360, crc_passed=True, validated_at=stamp(), rule_status=avg_meta['rule_status'], rule_confirmation_source=avg_meta['rule_confirmation_source'], one_model_one_forward_per_image=True, prediction_fusion=False, official_score=None, automatic_upload=False)
    write(WORK / 'submission_validation.json', validation)
    OUTPUT.with_suffix('.zip.sha256').write_text(sha(OUTPUT)+'  '+OUTPUT.name+'\n')
    avg_meta.update(official_submission_ready=True, full_prediction_validation=str(WORK / 'submission_validation.json'))
    write(WORK / 'average_validation.json', avg_meta)
    launch.update(stage='complete_validated', submission_validation=str(WORK / 'submission_validation.json'))
    write(WORK / 'launch.json', launch)
    decision = read(RESEARCH / 'research_and_decision.json')
    candidate = dict(method=validation['method'], source_epochs=[100,120,140,160], coefficients=[.25]*4, checkpoint=str(MODEL), path=str(OUTPUT), validation=str(WORK / 'submission_validation.json'), rule_status=avg_meta['rule_status'], official_score='pending user feedback')
    decision['last_chance_plan']['checkpoint_average_e100_e120_e140_e160_candidate'] = candidate
    decision['last_chance_plan']['next_candidate_priority'] = dict(method='e100/e120/e140/e160 equal parameter average', path=str(OUTPUT), status='generated and validated; pending user feedback')
    decision['updated_at'] = stamp()
    write(RESEARCH / 'research_and_decision.json', decision)
    print(json.dumps(validation, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
