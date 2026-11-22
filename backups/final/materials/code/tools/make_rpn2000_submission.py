"""Generate SC2 e130 predictions with 2000 RPN proposals using frozen code.

Only testing pre/post-NMS limits change. Both tasks use the same single forward
per image, with the original classifier fusion and detector postprocessing.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'backups/51.86-full150ep-e130-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def expand_test_proposals(model, topn):
    rpn = model.rpn
    before = dict(pre=dict(rpn._pre_nms_top_n), post=dict(rpn._post_nms_top_n), nms=rpn.nms_thresh)
    if before != dict(pre=dict(training=2000, testing=1000),
                      post=dict(training=2000, testing=1000), nms=0.7):
        raise ValueError(f'Unexpected original RPN configuration: {before}')
    rpn._pre_nms_top_n = {**rpn._pre_nms_top_n, 'testing': topn}
    rpn._post_nms_top_n = {**rpn._post_nms_top_n, 'testing': topn}
    if model.training or rpn.pre_nms_top_n() != topn or rpn.post_nms_top_n() != topn:
        raise AssertionError('Expanded proposal limits must be active in evaluation mode')
    return before, dict(pre=dict(rpn._pre_nms_top_n), post=dict(rpn._post_nms_top_n), nms=rpn.nms_thresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory-fraction', type=float, default=0.12)
    parser.add_argument('--images', type=Path,
                        default=Path.home() / 'cs701_data/cs701-sar-course-data/val')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'submissions/SC2_e130_val_shift-0.2_rpn2000.zip')
    parser.add_argument('--raw-out', type=Path,
                        default=ROOT / 'submissions/raw/SC2_e130_val_rpn2000_raw.pt')
    parser.add_argument('--record-dir', type=Path, default=ROOT / 'experiments/sc2_rpn2000')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1 or not 0 < args.memory_fraction <= 1:
        parser.error('Invalid inference resource settings')
    if args.out.exists() or args.raw_out.exists():
        parser.error('Output exists; choose new paths to preserve existing artifacts')
    args.record_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ.setdefault('HF_HOME', '/mnt/hdd/junkaichen/hf-cache')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this inference job')
    torch.cuda.set_device(0)  # physical GPU selected by CUDA_VISIBLE_DEVICES
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    sys.path.insert(0, str(BACKUP / 'code'))
    import predict
    import train
    import sarbench.model
    import sarbench.postprocess
    import sarbench.tiny
    for module in (predict, train, sarbench.model, sarbench.postprocess, sarbench.tiny):
        if not Path(module.__file__).resolve().is_relative_to((BACKUP / 'code').resolve()):
            raise RuntimeError(f'Must use frozen backup code: {module.__file__}')
    image_list = json.loads((args.images / 'images.json').read_text())['images']
    expected_ids = [im['id'] for im in image_list]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError('Duplicate image IDs')
    manifest = dict(checkpoint=str(BACKUP / 'weights/model.pt'),
                    checkpoint_sha256=sha256(BACKUP / 'weights/model.pt'),
                    run_args_sha256=sha256(BACKUP / 'weights/metrics.json'),
                    source_sha256={str(p.relative_to(BACKUP / 'code')): sha256(p)
                                   for p in sorted((BACKUP / 'code').rglob('*.py'))},
                    script_sha256=sha256(Path(__file__)), images=str(args.images),
                    images_sha256=sha256(args.images / 'images.json'),
                    expected_images=len(image_list), batch_size=args.batch_size,
                    workers=args.workers, threads=args.threads,
                    memory_fraction=args.memory_fraction,
                    physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                    inference='one 512x512 input and one model forward per image, bf16',
                    postprocessing={**predict.FINAL, 'shift': -0.2, 'cls_lambda': predict.LAM},
                    classification='original lambda=1 fusion from same 2000-proposal forward',
                    output=str(args.out), raw_output=str(args.raw_out),
                    image_count=0, model_forward_calls=0, proposal_counts=[])
    if manifest['checkpoint_sha256'] != '59a0309d30a80b101bd31f419e5e5884dd9a94536973ab7ba7959fe87e412c61':
        raise ValueError('Checkpoint does not match archived SC2 e130 SHA256')
    status_path = args.record_dir / 'status.json'
    started = time.monotonic()
    (args.record_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    status_path.write_text(json.dumps(dict(stage='loading', images=0), indent=2))
    original_load = predict.load_run

    def progress_hook(model, inputs, output):
        batch = inputs[0]
        if tuple(batch.shape[-2:]) != (512, 512):
            raise AssertionError('Unexpected input image size')
        logits, detections, _ = output
        if len(logits) != len(batch) or len(detections) != len(batch):
            raise AssertionError('Output count must match input images')
        counts = [len(item['raw_boxes']) for item in detections]
        if not all(0 < count <= 2000 for count in counts):
            raise AssertionError(f'Unexpected RPN proposal counts: {counts}')
        manifest['proposal_counts'].extend(counts)
        manifest['image_count'] += len(batch)
        manifest['model_forward_calls'] += 1
        if manifest['model_forward_calls'] == 1 or manifest['image_count'] % 100 < len(batch):
            torch.cuda.synchronize()
            state = dict(stage='inference', images=manifest['image_count'], total=len(image_list),
                         seconds=time.monotonic() - started,
                         peak_memory_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
            status_path.write_text(json.dumps(state, indent=2))
            print(f'PROGRESS {json.dumps(state)}', flush=True)

    def load_expanded(*load_args, **load_kwargs):
        model, saved = original_load(*load_args, **load_kwargs)
        before, after = expand_test_proposals(model, 2000)
        manifest['rpn_before'], manifest['rpn_after'] = before, after
        manifest['parameters'] = sum(p.numel() for p in model.parameters())
        manifest['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if manifest['parameters'] != 124866272 or manifest['trainable_parameters'] != 39656672:
            raise AssertionError('Model parameter counts changed')
        model.register_forward_hook(progress_hook)
        print(f'RPN_CONFIG {json.dumps(after)}', flush=True)
        return model, saved

    predict.load_run = load_expanded
    argv_before = sys.argv
    sys.argv = [str(BACKUP / 'code/predict.py'), '--run', str(BACKUP / 'weights'),
                '--weights', 'model.pt', '--images', str(args.images), '--shift', '-0.2',
                '--batch-size', str(args.batch_size), '--workers', str(args.workers),
                '--out', str(args.out), '--raw-out', str(args.raw_out)]
    manifest['predict_argv'] = sys.argv[1:]
    try:
        predict.main()
        if manifest['image_count'] != len(image_list) or max(manifest['proposal_counts']) <= 1000:
            raise AssertionError('Incomplete inference or expanded RPN limit was not exercised')
        with zipfile.ZipFile(args.out) as archive:
            if archive.testzip() is not None:
                raise AssertionError('Corrupt output ZIP')
            manifest['detections'] = len(json.loads(archive.read('detection.json')))
        manifest['seconds'] = time.monotonic() - started
        manifest['peak_memory_gib'] = torch.cuda.max_memory_allocated() / 1024 ** 3
        manifest['output_sha256'] = sha256(args.out)
        manifest['raw_sha256'] = sha256(args.raw_out)
        counts = manifest.pop('proposal_counts')
        manifest['proposal_count_min'], manifest['proposal_count_max'] = min(counts), max(counts)
        args.out.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
        args.out.with_suffix('.sha256').write_text(f'{manifest["output_sha256"]}  {args.out.name}\n')
        (args.record_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        status_path.write_text(json.dumps(dict(stage='complete', images=len(image_list), output=str(args.out)), indent=2))
    except BaseException as error:
        status_path.write_text(json.dumps(dict(stage='failed', images=manifest['image_count'],
                                               error=f'{type(error).__name__}: {error}'), indent=2))
        raise
    finally:
        sys.argv = argv_before
        predict.load_run = original_load


if __name__ == '__main__':
    main()
