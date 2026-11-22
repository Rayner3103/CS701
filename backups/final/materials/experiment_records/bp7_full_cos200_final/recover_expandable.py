import fcntl, hashlib, json, shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()

def main():
 root=Path(__file__).resolve().parents[2]
 old=root.parent/'cs701_runs/BP7_full9392_cos200_stop160_s0_20261031'
 new=root.parent/'cs701_runs/BP7_full9392_cos200_stop160_s0_20261031_expandable'
 if new.exists(): raise RuntimeError('Recovery suite exists')
 with (old/'driver.lock').open('r') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status=json.loads((old/'status.json').read_text())
  if status['stage']!='failed': raise RuntimeError('Source must be failed')
  sources={}
  for line in (old/'source.sha256').read_text().splitlines():
   expected,name=line.split('  ',1);p=(old/'code'/name).resolve()
   if not p.is_relative_to((old/'code').resolve()) or sha(p)!=expected: raise ValueError(name)
   sources[name]=p
  for name in ['tools/run_bp7_cos200_final.py','tests/test_cos200_final.py',str(Path(__file__).resolve().relative_to(root))]: sources[name]=root/name
  manifest=json.loads((old/'launch_manifest.json').read_text())
  if any(Path(v).exists() for v in manifest['submission_paths'].values()): raise RuntimeError('Output exists')
  checkpoint=old/'full_train/training_state.pt'
  recovery=dict(source_suite=str(old),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),epoch=1,global_step=587,allocator_config='expandable_segments:True')
  (new/'code').mkdir(parents=True)
  for name,p in sorted(sources.items()):
   target=new/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  (new/'source.sha256').write_text(''.join(f'{sha(new/"code"/name)}  {name}\n' for name in sorted(sources)))
  manifest.update(created_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),source_files=len(sources),recovery=recovery,execution='Recover e1 with expandable CUDA segments; original model, optimizer, LR, batch, workers and 20 percent cap')
  (new/'launch_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
  (new/'recovery.json').write_text(json.dumps(dict(recovery=recovery,source_status=status,source_log_sha256=sha(old/'full_train/stdout.log'),original_preserved=True),indent=2,ensure_ascii=False)+'\n')
  print(json.dumps(dict(suite=str(new),recovery=recovery),indent=2))

if __name__=='__main__': main()
