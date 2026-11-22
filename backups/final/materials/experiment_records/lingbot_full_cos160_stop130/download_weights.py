from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, json
from huggingface_hub import hf_hub_download
repo='robbyant/lingbot-vision-vit-base'
revision='f606f8c6c4002234ea68038f4d7c7cf57da96dfa'
expected='783dfb59014c34e9f0013db60bf6f5cc3c6b0604eb9528fb5df65e80769c5825'
p=Path(hf_hub_download(repo, 'model.pt', revision=revision, cache_dir='/mnt/hdd/junkaichen/hf-cache/hub'))
h=hashlib.sha256()
with p.open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==expected and p.stat().st_size==342852282
record=dict(time=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),repo=repo,revision=revision,filename='model.pt',path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size,source='https://huggingface.co/'+repo+'/tree/'+revision,license='Apache-2.0')
Path(__file__).with_name('pretrained.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
