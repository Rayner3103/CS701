#!/usr/bin/env python3
"""Restore the final val/test weights from GitHub Release assets, or check local files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify all files without downloading.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    for record in manifest['files']:
        target = root / record['path']
        if not target.exists():
            if args.check or 'download_url' not in record:
                raise SystemExit(f'Missing file: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            print(f'Downloading {record["path"]}', flush=True)
            try:
                request = urllib.request.Request(record['download_url'], headers={'User-Agent': 'CS701-checkpoints'})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
                    while block := response.read(8 * 1024 * 1024):
                        out.write(block)
                if partial.stat().st_size != record['bytes'] or sha256(partial) != record['sha256']:
                    raise SystemExit(f'Download verification failed: {record["path"]}')
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
        if target.stat().st_size != record['bytes'] or sha256(target) != record['sha256']:
            raise SystemExit(f'File verification failed: {record["path"]}')
        print(f'OK: {record["path"]}', flush=True)


if __name__ == '__main__':
    main()
