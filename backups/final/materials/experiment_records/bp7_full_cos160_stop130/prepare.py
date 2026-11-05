"""Prepare this authorized, single full-data trial; does not start training."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[2]
runs = root.parent / 'cs701_runs'
suite = runs / 'BP7_full9392_cos160_stop130_s0_20261031'
baseline = runs / 'BP7_LR130_local_s0_20261029_v2'
source_manifest = runs / 'BP7_fixedLR160_local_s0_20261030_direct/source.sha256'
if suite.exists():
    raise SystemExit('Suite already exists; inspect before any further action')
names = {line.split('  ', 1)[1] for line in source_manifest.read_text().splitlines()}
names.update({'tools/run_bp7_full_stop130.py', 'tests/test_full_stop130.py'})
for name in names:
    if not (root / name).is_file():
        raise SystemExit('Missing source ' + name)
(suite / 'code').mkdir(parents=True)
for name in sorted(names):
    dest = suite / 'code' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / name, dest)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
(suite / 'source.sha256').write_text(''.join(f'{sha(suite / "code" / name)}  {name}\n' for name in sorted(names)))
old = json.loads((baseline / 'launch_manifest.json').read_text())
manifest = dict(
    created_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),
    authorization='User accepted original BP7 full9392, original peak LR, cosine160 and stop/export at130. Subsequent scheduler question answered without changing the fixed plan.',
    source=str(root), git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    source_files=len(names), dependency_suite=str(baseline), data=old['data'], data_sha256=old['data_sha256'],
    official_data=str(root.parent / 'cs701_data/cs701-sar-course-data'),
    expected_parameters=125076960, expected_trainable_parameters=39867360, expected_floating_buffers=16,
    policy_margin=1.0, policy_margin_note='Required by shared snapshot validator only; no dynamic candidate selection in this run.',
    architecture='Original BP7 DINOv3 LoRA80 alpha160 Cascade1536 SpatialPrior64, roi_context=0',
    epochs=160, schedule_epochs=160, stop_after_epoch=130, head_lr=1e-4, backbone_lr=3e-4, warmup_epochs=1, seed=0,
    training_split='train+val+test', training_images=9392, eval_splits='', steps_per_epoch=587, expected_updates_at_stop=76310,
    initialization='Fresh generic DINOv3 pretrained; task modules and LoRA initialized afresh; smoke weights discarded.',
    resources=dict(gpus=[0, 1], cpu_affinity=[0, 1, 2, 3], nice=10, global_batch=16, workers_per_rank=2,
                   threads_per_rank=2, prediction_batch=2, prediction_workers=0),
    prediction_shift=-0.2, submission_path=str(root / 'submissions/BP7_full_cos160_e130.zip'),
    failure_policy='Stop and preserve logs/checkpoints, no blind retries.', maximum_full_runs=1,
    automatic_continue_after_epoch130=False, official_submissions=0,
    comparison_targets=dict(SC2_full150_e130_delta_m=51.86, BP7_full130_delta_m=51.76),
    prior_local_e130_official_comparison=dict(cos130_delta_m=45.83, cos160_delta_m=47.12,
                                            caution='Single seed, not a demonstrated full-data improvement.'),
    smoke=dict(training_images=48, epochs=160, schedule_epochs=160, stop_after_epoch=1, workers=0, inference_images=8),
)
(suite / 'launch_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
shutil.copy2(root / 'experiments/bp7_full_cos160_stop130/README.md', suite / 'README.md')
print(json.dumps(dict(suite=str(suite), source_files=len(names), submission_path=manifest['submission_path']), indent=2))
