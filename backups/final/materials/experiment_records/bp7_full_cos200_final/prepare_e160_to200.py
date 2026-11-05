"""Freeze the unchanged, complete e160 BP7 trajectory for its e200 finish."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    source = root.parent / 'cs701_runs/BP7_full9392_cos200_e147to160_s0_20261101_tmp'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', default='', choices=['', '_v2'])
    suffix = parser.parse_args().suffix
    suite = root.parent / ('cs701_runs/BP7_full9392_cos200_e160to200_s0_20261101' + suffix)
    if suite.exists():
        raise FileExistsError(f'Refusing to overwrite {suite}')
    if json.loads((source / 'status.json').read_text())['stage'] != 'complete':
        raise ValueError('The e160 source must be complete')
    checkpoint = source / 'full_train_e160/training_state.pt'
    state_hash = '8d4b5577d600aef6b7225bc1f103b36ee160a747ba4e8f288e5aa8c1c7d3c1dd'
    if sha256(checkpoint) != state_hash:
        raise ValueError('The verified e160 full state changed')
    previous_zip = root / 'submissions/BP7_full_cos200_e160.zip'
    zip_hash = '6df4c1cc49971afffff1a78da76a3533ea27031ec43732feee6e324ac190ed98'
    if sha256(previous_zip) != zip_hash:
        raise ValueError('The scored e160 submission changed')
    sources = {}
    for line in (source / 'source.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (source / 'code' / name).resolve()
        if name in sources or not path.is_relative_to((source / 'code').resolve()) or sha256(path) != expected:
            raise ValueError(f'Frozen source mismatch: {name}')
        sources[name] = path
    additions = ('tools/run_bp7_e160_to200.py', 'tests/test_e160_to200.py',
                 'experiments/bp7_full_cos200_final/prepare_e160_to200.py')
    for name in additions:
        if name in sources or not (root / name).is_file():
            raise ValueError(f'Expected a new source file: {name}')
        sources[name] = root / name
    outputs = {str(epoch): str(root / f'submissions/BP7_full_cos200_e{epoch}.zip')
               for epoch in (180, 200)}
    for output in map(Path, outputs.values()):
        if output.exists() or output.with_suffix('.sha256').exists():
            raise FileExistsError(f'Submission already exists: {output}')
    for name, path in sorted(sources.items()):
        destination = suite / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (suite / 'source.sha256').write_text(''.join(
        f'{sha256(suite / "code" / name)}  {name}\n' for name in sorted(sources)))
    previous = json.loads((source / 'launch_manifest.json').read_text())
    manifest = dict(
        created_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),
        source_suite=str(source), source_epoch=160, source_checkpoint_sha256=state_hash,
        data=previous['data'], official_data=previous['official_data'],
        existing_e160_submission=str(previous_zip), existing_e160_submission_sha256=zip_hash,
        submission_paths=outputs, source_files=len(sources),
        gpu_memory_fraction=0.35, allocator_config='expandable_segments:True',
        temp_dir=str(root / '.runtime/bp7_e160'),
        epochs=200, schedule_epochs=200, stop_after_epoch=200,
        training_images=9392, global_batch=16, save_every=10,
        expected_final_global_step=117400, export_epochs=[180, 200],
        remaining_submissions_user_reported=8, best_reported_delta_m=53.41,
        authorization='Continue the user-authorized autonomous BP7 optimization; keep architecture and peak learning rates unchanged, LingBot paused.',
        decision='Finish the existing cosine-200 trajectory, preserving e160 and its official best score; one training job, no restart of the cosine schedule.',
        prepared_not_launched=True, automatic_upload=False,
        watchdog_no_progress_seconds=300)
    (suite / 'launch_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(suite=str(suite), source_files=len(sources), submissions=outputs), indent=2))


if __name__ == '__main__':
    main()
