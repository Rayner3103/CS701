"""Freeze the authorized LingBot trial; the controller waits for BP7 before GPU work."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, json, shutil, subprocess

root=Path(__file__).resolve().parents[2]
meta=Path(__file__).resolve().parent
runs=root.parent/'cs701_runs'
previous=runs/'BP7_full9392_cos160_stop130_s0_20261031'
suite=runs/'LingBotB_full9392_cos160_stop130_s0_20261031'
if suite.exists(): raise SystemExit('Suite exists; refusing to overwrite')
names={line.split('  ',1)[1] for line in (previous/'source.sha256').read_text().splitlines()}
names.update({'sarbench/lingbot.py','tools/run_lingbot_full_stop130.py','tests/test_lingbot.py','tests/test_lingbot_queue.py'})
names.update(str(p.relative_to(root)) for p in (root/'sarbench/vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
for name in names:
 if not (root/name).is_file(): raise SystemExit('Missing source: '+name)
budget=json.loads((meta/'cpu_integration_validation.json').read_text())['budget']
pretrained=json.loads((meta/'pretrained.json').read_text())
old=json.loads((previous/'launch_manifest.json').read_text())
(suite/'code').mkdir(parents=True)
for name in sorted(names):
 target=suite/'code'/name
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(root/name,target)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(suite/'source.sha256').write_text(''.join(f'{sha(suite/"code"/name)}  {name}\n' for name in sorted(names)))
manifest={**old,
 'created_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),
 'authorization':'User: after this current experiment ends, try LingBot-B. Sequential single-trial comparison; current BP7 completes/export first.',
 'source':str(root),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_files':len(names),'dependency_suite':str(previous),'wait_for_suite':str(previous),
 'backbone':'lingbot_b','architecture':'LingBot-Vision-B/16 LoRA80 alpha160 Cascade1536 SpatialPrior64, no context',
 'expected_parameters':budget['parameters'],'expected_trainable_parameters':budget['trainable_parameters'],
 'expected_floating_buffers':budget['floating_buffers'],'pretrained':pretrained,
 'submission_path':str(root/'submissions/LingBotB_full_cos160_e130.zip'),
 'initialization':'Fresh pinned generic LingBot-Vision-Base pretrained backbone; no BP7 task weights; smoke weights discarded.',
 'backbone_comparison':'Only backbone integration changes; identical training script, BP7 heads, data, peak LR, 160-epoch cosine and e130 stop.',
 'input_adaptation':'RGB patch kernels summed for grayscale; same SAR mean/std, 512x512 single image.',
 'rope_policy':'Retain upstream RoPE; disable extra random coordinate rescaling in downstream training to match BP7 augmentation recipe.',
 'cpu_checks':{'backbone_tests':9,'queue_tests':18,'existing_adapter_tests':5,'real_weight_forward_and_gradients':'passed'},
 'gpu_smoke_policy':'Only after BP7 completion, e130 export validation and dependency lock release; stop if smoke fails.',
 'rules_source':'https://web-production-c241e.up.railway.app/docs#terms',
 'rules_public_asset_sha256':'3651c33ae12387243723f7065efb3e2e5589c850030254e9de001528f79d8fbc',
 'upstream_source':json.loads((root/'sarbench/vendor/lingbot_vision/UPSTREAM.json').read_text()),
 'comparison_targets':{'paired_model':str(previous),'SC2_historical_delta_m':51.86,'BP7_historical_delta_m':51.76},
}
for key in ('policy_margin','policy_margin_note','prior_local_e130_official_comparison'):
 manifest.pop(key,None)
(suite/'launch_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
for name in ('README.md','pretrained.json','cpu_integration_validation.json','validate_integration.py'):
 shutil.copy2(meta/name,suite/name)
print(json.dumps({'suite':str(suite),'source_files':len(names),'budget':budget,'submission_path':manifest['submission_path']},indent=2))
