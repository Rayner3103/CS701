"""Keep the standalone trainer attached to an exit-status record and a local log."""
import argparse
import ctypes
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_safety import atomic_json


def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job-dir',type=Path,required=True)
    parser.add_argument('training_args',nargs=argparse.REMAINDER);opts=parser.parse_args()
    arguments=opts.training_args[1:] if opts.training_args[:1]==['--'] else opts.training_args
    opts.job_dir.mkdir(parents=True,exist_ok=True)
    status=opts.job_dir/'job_status.json'
    if status.exists():raise FileExistsError('Use a new job directory')
    command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('run_project.py')),*arguments]
    state={'status':'STARTING','started_utc':now(),'supervisor_pid':__import__('os').getpid(),'command':command}
    atomic_json(status,state)
    awake=ctypes.windll.kernel32.SetThreadExecutionState if sys.platform=='win32' else None
    if awake:state['wake_request']=bool(awake(0x80000001))
    child=None
    try:
        with (opts.job_dir/'stdout.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=Path(__file__).resolve().parent,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='RUNNING',training_pid=child.pid);atomic_json(status,state)
            code=child.wait()
        state.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,finished_utc=now())
        atomic_json(status,state);return code
    except BaseException as exc:
        if child and child.poll() is None:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=False,timeout=15)
        state.update(status='SUPERVISOR_ERROR',error=repr(exc),finished_utc=now());atomic_json(status,state)
        raise
    finally:
        if awake:awake(0x80000000)


if __name__=='__main__':raise SystemExit(main())
