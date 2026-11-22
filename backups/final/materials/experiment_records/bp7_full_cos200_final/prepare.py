"""Freeze one cos200 candidate; preserve all existing experiment snapshots."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    previous = root.parent / 'cs701_runs/BP7_full9392_cos160_stop130_s0_20261031'
    suite = root.parent / 'cs701_runs/BP7_full9392_cos200_stop160_s0_20261031'
    if suite.exists():
        raise RuntimeError('Suite already exists')
    if json.loads((previous / 'status.json').read_text())['stage'] != 'complete':
        raise RuntimeError('Source BP7 run must be complete')
    sources = {}
    for line in (previous / 'source.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        source = (previous / 'code' / name).resolve()
        if not source.is_relative_to((previous / 'code').resolve()) or sha256(source) != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
        sources[name] = source
    for name in ['tools/run_bp7_cos200_final.py', 'tools/train_gpu_limited.py',
                 'tests/test_cos200_final.py', str(Path(__file__).resolve().relative_to(root))]:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
        sources[name] = root / name
    outputs = {str(e): str(root / f'submissions/BP7_full_cos200_e{e}.zip') for e in (130, 160)}
    if any(Path(p).exists() for p in outputs.values()):
        raise RuntimeError('Submission already exists')
    (suite / 'code').mkdir(parents=True)
    for name, source in sorted(sources.items()):
        target = suite / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (suite / 'source.sha256').write_text(''.join(
        f'{sha256(suite / "code" / name)}  {name}\n' for name in sorted(sources)))
    manifest = json.loads((previous / 'launch_manifest.json').read_text())
    manifest.update(
        created_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),
        authorization='User: skip control groups; prioritize the best candidate and increase GPU use with one day left.',
        source_files=len(sources), dependency_suite=str(previous),
        epochs=200, schedule_epochs=200, stop_after_epoch=160,
        export_epochs=[130, 160], expected_updates_at_exports={'130': 76310, '160': 93920},
        expected_updates_at_stop=93920, submission_paths=outputs,
        automatic_continue_after_epoch130=True, compile=False, gpu_memory_fraction=0.20,
        execution='Eager blocks, gradient checkpointing, parallel with LingBot after a memory-limited smoke test',
        decision='Keep BP7 and peak LR; slower cosine, export e130 and e160 on one trajectory',
        comparison_targets={'best_reported_project_delta_m': 52.78})
    manifest.pop('submission_path', None)
    manifest.pop('prior_local_e130_official_comparison', None)
    manifest['resources'].update(workers_per_rank=1, threads_per_rank=1, gpu_memory_fraction=0.20)
    manifest['smoke'].update(epochs=200, schedule_epochs=200)
    (suite / 'launch_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(suite=str(suite), source_files=len(sources), submissions=outputs), indent=2))


if __name__ == '__main__':
    main()
