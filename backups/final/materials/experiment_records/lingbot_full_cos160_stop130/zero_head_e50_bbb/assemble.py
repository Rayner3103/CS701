"""Construct a zero-adaptation LingBot plus BP7 head checkpoint on the CPU."""
import gc
import json
from pathlib import Path


def assemble(root, run, manifest):
    import torch
    import train
    from sarbench.adapters import LoRA
    from sarbench.training_state import seed_everything
    from run_bp7_schedule_search import read_json, sha256_file, stamp, write_json
    torch.set_num_threads(1)
    donor = root.parent / 'cs701_runs/BP7_full9392_cos160_stop130_s0_20261031/full_train'
    source = donor / 'ckpt_e50.pt'
    source_digest = sha256_file(source)
    paired = read_json(root / 'experiments/bp7_full_cos160_stop130/submission_e50_aaa/validation.json')
    assert paired['checkpoint_sha256'] == source_digest and paired['checkpoint_epoch'] == 50
    saved = json.loads((donor / 'log.txt').read_text().splitlines()[0])
    assert saved['train_split'] == 'train+val+test' and saved['schedule_epochs'] == 160
    saved.update(backbone='lingbot_b', out=str(run.parent), name=run.name, compile=False, resume=None)
    args = train.parse_args(['--backbone', 'lingbot_b', '--init', 'pretrained', '--adapt', 'lora'])
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, Path(value) if key in ('data', 'out') else value)
    seed_everything(0)
    model = train.build_model(args).eval()
    donor_state = torch.load(source, map_location='cpu', weights_only=True)
    heads = {key.replace('._orig_mod', ''): tensor for key, tensor in donor_state.items() if not key.startswith('backbone.')}
    current = model.state_dict()
    expected = {key for key in current if not key.startswith('backbone.')}
    assert set(heads) == expected and len(heads) == 88
    assert all(current[key].shape == tensor.shape for key, tensor in heads.items())
    missing, unexpected = model.load_state_dict(heads, strict=False)
    assert not unexpected and all(key.startswith('backbone.') for key in missing)
    assert all(torch.equal(model.state_dict()[key], tensor) for key, tensor in heads.items())
    adapters = [module for module in model.backbone.modules() if isinstance(module, LoRA)]
    assert len(adapters) == 48 and all(torch.count_nonzero(module.up.weight) == 0 for module in adapters)
    budget = dict(parameters=sum(p.numel() for p in model.parameters()),
                  floating_buffers=sum(b.numel() for b in model.buffers() if b.is_floating_point()),
                  configured_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert tuple(budget.values()) == (125105376, 27664, 39867360)
    assert all(torch.isfinite(tensor).all() for tensor in model.state_dict().values())
    published = torch.load(manifest['pretrained']['path'], map_location='cpu', weights_only=True)['model']
    published = {key.removeprefix('backbone.'): value for key, value in published.items()}
    published['patch_embed.proj.weight'] = published['patch_embed.proj.weight'].sum(1, keepdim=True)
    base_state = {key.removeprefix('vit.').replace('.base.', '.'): value
                  for key, value in model.backbone.state_dict().items()
                  if '.down.' not in key and '.up.' not in key}
    assert set(base_state) == set(published)
    assert all(torch.equal(value, published[key]) for key, value in base_state.items())
    model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    run.mkdir()
    (run / 'log.txt').write_text(json.dumps(saved) + '\nINFERENCE ASSEMBLY ONLY: architecture and donor recipe; zero new training steps; head donor epoch50.\n')
    checkpoint = run / 'model.pt'
    torch.save(model.state_dict(), checkpoint)
    digest = sha256_file(checkpoint)
    assembly = dict(time=stamp(), new_training_steps=0, new_optimizer_steps=0,
                    head_source_checkpoint=str(source), head_source_sha256=source_digest,
                    head_source_epoch=50, head_source_training_images=9392, copied_head_tensors=88,
                    copied_head_parameters=sum(t.numel() for t in heads.values()),
                    all_heads_identical_to_aaa=True, pretrained_backbone_tensors_unchanged=True,
                    lora_up_all_zero=True, copied_dinov3_backbone_tensors=0,
                    weights=str(checkpoint), weights_sha256=digest, budget=budget,
                    paired_submission=str(root / 'submissions/aaa.zip'), paired_submission_sha256=paired['sha256'],
                    args_note='Architecture reconstruction and donor recipe only; no optimization after splicing.')
    write_json(run.parent / 'assembly.json', assembly)
    del model, current, heads, donor_state, published, base_state, adapters
    gc.collect()
    return checkpoint, digest, paired
