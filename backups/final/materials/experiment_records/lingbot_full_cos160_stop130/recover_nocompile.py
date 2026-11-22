"""Preserve the failed preflight and freeze one eager-mode recovery attempt."""
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    runs = root.parent / 'cs701_runs'
    failed = runs / 'LingBotB_full9392_cos160_stop130_s0_20261031'
    suite = runs / 'LingBotB_full9392_cos160_stop130_s0_20261031_nocompile'
    with (failed / 'driver.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads((failed / 'status.json').read_text())
        if status['stage'] != 'failed' or (failed / 'full_train').exists():
            raise RuntimeError('Recovery is only for the failed preflight, before full training')
        if list((failed / 'validation').rglob('*.pt')):
            raise RuntimeError('Unexpected completed smoke checkpoint; inspect before recovery')
        if suite.exists():
            raise RuntimeError('Recovery suite exists; refusing to overwrite')
        entries = {}
        for line in (failed / 'source.sha256').read_text().splitlines():
            expected, name = line.split('  ', 1)
            source = (failed / 'code' / name).resolve()
            if not source.is_relative_to((failed / 'code').resolve()) or sha256(source) != expected:
                raise ValueError(f'Failed snapshot checksum mismatch: {name}')
            entries[name] = source
        manifest = json.loads((failed / 'launch_manifest.json').read_text())
        if Path(manifest['submission_path']).exists():
            raise RuntimeError('Refusing to overwrite an existing LingBot submission')
        # Preserve the exact failed source except the launcher and its regression test.
        changed = ['tools/run_lingbot_full_stop130.py', 'tests/test_lingbot_queue.py']
        for name in changed:
            entries[name] = root / name
        entries[str(Path(__file__).resolve().relative_to(root))] = Path(__file__).resolve()
        (suite / 'code').mkdir(parents=True)
        for name, source in sorted(entries.items()):
            target = suite / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (suite / 'source.sha256').write_text(''.join(
            f'{sha256(suite / "code" / name)}  {name}\n' for name in sorted(entries)))
        recovery = dict(
            created_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),
            failed_suite=str(failed), failed_status=status,
            failure='CUDA AOTAutograd plus external block checkpointing: recomputed tensor metadata mismatch',
            fix='Remove --compile only; retain gradient checkpointing and safety checks',
            changed_existing_sources=changed, training_started_before_recovery=False,
            failed_smoke_log_sha256=sha256(failed / 'validation/smoke_train.log'),
        )
        manifest.update(
            created_at=recovery['created_at'], source_files=len(entries), compile=False,
            recovery=recovery,
            backbone_comparison='LingBot backbone and eager execution; identical training script, heads, data, learning rates, batch, cosine schedule and stop epoch',
        )
        for name, value in [('launch_manifest.json', manifest), ('recovery.json', recovery)]:
            (suite / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        for name in ['README.md', 'pretrained.json', 'cpu_integration_validation.json', 'validate_integration.py']:
            shutil.copy2(failed / name, suite / name)
        with (suite / 'README.md').open('a') as stream:
            stream.write('\nRecovery: the original CUDA compiled smoke failed before its first optimizer update. '
                         'This suite disables block compilation only and repeats the complete smoke gate. '
                         'Original failure and source snapshot are preserved; see recovery.json.\n')
        print(json.dumps(dict(suite=str(suite), changed=changed, source_files=len(entries)), indent=2))


if __name__ == '__main__':
    main()
