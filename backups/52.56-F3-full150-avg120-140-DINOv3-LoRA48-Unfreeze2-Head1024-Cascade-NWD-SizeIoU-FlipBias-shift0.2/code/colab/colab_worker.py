"""Colab GPU worker: runs the jobs of colab/queue.json one by one and publishes their logs through git.

Claude (on the local machine) only appends jobs to colab/queue.json and pushes; this worker only writes results/<job>/
and colab/state/<job>.json, so the two sides never edit the same file. runs/ is a symlink into Google Drive, so
checkpoints survive a disconnected runtime: an interrupted job is restarted with --resume (it continues from last.pt).

python scripts/colab_worker.py [--worker A] [--idle-minutes 20] [--sync-minutes 10]
Job fields: name, cmd (list, run from the repo root), run_dir, max_hours (optional), collect (optional extra files),
worker (optional, default 'A'): several Colab runtimes run in parallel, each only takes the jobs of its own lane.
When no job is left the worker waits --idle-minutes for new ones, then releases the Colab GPU to stop billing.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE, STATE, RESULTS = ROOT / 'colab/queue.json', ROOT / 'colab/state', ROOT / 'results'
MAX_ATTEMPTS = 4  # a failed job is resumed this many times in total before it is marked failed
LOGS = ('config.json', 'history.jsonl', 'train_log.jsonl', 'summary.json', 'hardware.json', 'parameter_audit.json')


def git(*args, check=True):
    return subprocess.run(['git', *args], cwd=ROOT, check=check, capture_output=True, text=True)


def sync(message):
    """Commit results/ and colab/state/ and push; rebase onto Claude's new jobs first (disjoint files)."""
    paths = [p for p in ('results', 'colab/state') if (ROOT / p).exists() and any((ROOT / p).rglob('*'))]
    if paths:  # git add fails on a pathspec with no files yet (first sync of a fresh clone)
        git('add', '-A', '--', *paths)
    if git('diff', '--cached', '--quiet', check=False).returncode:
        git('commit', '-m', message)
    for _ in range(3):
        git('pull', '--rebase', '-q', check=False)
        if git('push', '-q', check=False).returncode == 0:
            return
        time.sleep(10)
    print('warning: push failed; will retry at the next sync', flush=True)


def state(name):
    path = STATE / f'{name}.json'
    return json.loads(path.read_text()) if path.exists() else {}


def write_state(name, **fields):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f'{name}.json').write_text(json.dumps({**state(name), **fields}, indent=2))


def collect(job):
    try:
        _collect(job)
    except OSError as e:  # the Drive mount can drop for a moment; the next sync copies the logs again
        print('warning: collect failed:', e, flush=True)


def _collect(job):
    out = RESULTS / job['name']
    out.mkdir(parents=True, exist_ok=True)
    run = ROOT / job['run_dir']
    for name in LOGS:
        if (run / name).exists():
            shutil.copy2(run / name, out / name)
    for extra in job.get('collect', []):
        if (ROOT / extra).exists():
            shutil.copy2(ROOT / extra, out / Path(extra).name)


def last_epoch(job):
    log = ROOT / job['run_dir'] / 'train_log.jsonl'
    if not log.exists():
        log = ROOT / job['run_dir'] / 'history.jsonl'
    if not log.exists():
        return 0
    try:
        lines = [l for l in log.read_text().splitlines() if l.strip()]
    except OSError:
        return 0
    return json.loads(lines[-1]).get('epoch', 0) if lines else 0


def gpu_name():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return 'no GPU'


def ready_jobs(jobs, worker):
    return [job for job in jobs if job.get('enabled', True) and job.get('worker', 'A') == worker
            and state(job['name']).get('status') in (None, 'running')
            and all(state(required).get('status') == 'finished' for required in job.get('requires', []))]


def run(job, sync_minutes, worker):
    previous = state(job['name'])
    cmd = list(job['cmd'])
    if previous.get('status') == 'running' and '--resume' not in cmd and job.get('resumable', True):
        cmd.append('--resume')  # the runtime died mid-job; last.pt on Drive holds the progress
    write_state(job['name'], status='running', worker=worker, started=time.strftime('%Y-%m-%d %H:%M:%S'), gpu=gpu_name(),
                attempts=previous.get('attempts', 0) + 1, cmd=cmd)
    sync(f"start {job['name']}")
    log_path = ROOT / job['run_dir'] / 'worker.stdout.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0, last_sync = time.time(), time.time()
    with log_path.open('a') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        timeout = None
        while proc.poll() is None:
            time.sleep(30)
            if job.get('max_hours') and time.time() - t0 > job['max_hours'] * 3600:
                timeout = f"timeout after {job['max_hours']} h"
                proc.terminate()
                proc.wait(60)
            if time.time() - last_sync > sync_minutes * 60:
                collect(job)
                write_state(job['name'], epoch=last_epoch(job), updated=time.strftime('%Y-%m-%d %H:%M:%S'))
                sync(f"progress {job['name']} epoch {last_epoch(job)}")
                last_sync = time.time()
    collect(job)
    try:
        tail = log_path.read_text().splitlines()[-30:]
        (RESULTS / job['name']).mkdir(parents=True, exist_ok=True)
        (RESULTS / job['name'] / 'stdout_tail.txt').write_text('\n'.join(tail))
    except OSError:
        pass
    status = timeout or ('finished' if proc.returncode == 0 else f'failed({proc.returncode})')
    attempts = previous.get('attempts', 0) + 1
    if proc.returncode and not timeout and attempts < MAX_ATTEMPTS and job.get('resumable', True):
        # e.g. a transient Drive I/O error: keep it 'running' so the main loop restarts it with --resume
        write_state(job['name'], status='running', last_error=status, epoch=last_epoch(job))
        sync(f"{status} {job['name']}, retrying with --resume")
        print(job['name'], status, 'retrying', flush=True)
        time.sleep(60)
        return
    write_state(job['name'], status=status, finished=time.strftime('%Y-%m-%d %H:%M:%S'), epoch=last_epoch(job),
                hours=round((time.time() - t0) / 3600, 2))
    sync(f"{status} {job['name']}")
    print(job['name'], status, flush=True)


def main():
    global QUEUE
    p = argparse.ArgumentParser()
    p.add_argument('--worker', default='A', help='lane: only jobs whose "worker" field matches (default A)')
    p.add_argument('--queue', default='colab/queue.json', help='queue JSON, relative to the repo root')
    p.add_argument('--idle-minutes', type=float, default=20)
    p.add_argument('--sync-minutes', type=float, default=10)
    p.add_argument('--no-release', action='store_true', help='keep the runtime when the queue is empty')
    a = p.parse_args()
    QUEUE = ROOT / a.queue
    idle_since = None
    while True:
        git('pull', '--rebase', '-q', check=False)
        jobs = json.loads(QUEUE.read_text()) if QUEUE.exists() else []
        todo = ready_jobs(jobs, a.worker)
        if todo:
            idle_since = None
            run(todo[0], a.sync_minutes, a.worker)
            continue
        idle_since = idle_since or time.time()
        if time.time() - idle_since > a.idle_minutes * 60:
            print(f'queue of worker {a.worker} empty: stopping', flush=True)
            if not a.no_release:
                try:
                    from google.colab import runtime
                    runtime.unassign()  # release the GPU so compute units stop draining
                except ImportError:
                    pass
            return 0
        time.sleep(60)


if __name__ == '__main__':
    sys.exit(main())
