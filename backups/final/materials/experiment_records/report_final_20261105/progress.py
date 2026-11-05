#!/usr/bin/env python3
"""Experiment progress table for the report runs: python3 progress.py"""
import re, time, sys
from pathlib import Path
E = Path(__file__).resolve().parent; R = Path('/mnt/hdd/junkaichen/CS701/cs701_runs')
V = ['base', 'noSP', 'noSizeIoU', 'noFlipBias', 'noNWD', 'noCascade', 'noSmallAnchors', 'noTrainNorms', 'smallCapacity']
jobs = [('RPT_TWIN_cos200_s1', 180), ('RPT_TWIN_cos200_s2', 180)] + [(f'RPT_LOO36_{v}_s0', 36) for v in V] + [('RPT_REF24_s0', 24)] + [(f'RPT_LOO36_{v}_s{s}', 36) for s in (1, 2) for v in V]
q = (E / 'logs/queue.log').read_text() if (E / 'logs/queue.log').exists() else ''
print(f"{'run':34s} {'status':10s} {'epoch':>9s}")
done = run = 0
for n, tot in jobs:
    lg = R / n / 'run/log.txt'; ep = 0
    if lg.exists():
        m = re.findall(r'^epoch (\d+) iter (\d+)/(\d+)', lg.read_text(errors='ignore'), re.M)
        if m: ep = int(m[-1][0]) - (0 if m[-1][1] == m[-1][2] else 1)
    st = 'done' if f'done {n} ok=True' in q else 'failed' if f'done {n} ok=False' in q else 'running' if lg.exists() else 'queued'
    done += st == 'done'; run += st == 'running'
    print(f"{n:34s} {st:10s} {ep:4d}/{tot:<4d}")
print(f"{done}/{len(jobs)} done, {run} running, {time.strftime('%F %H:%M')}")
