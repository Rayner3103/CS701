"""CPU checks for the full-data e130 handoff and submission validation."""
import copy
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_bp7_full_stop130 as full
import train


def parse(command):
    return train.parse_args(command[command.index('train.py') + 1:])


def test_full_run_keeps_160_cosine_but_stops_at_130(tmp_path):
    command = full.training_command('/python', tmp_path / 'data', tmp_path / 'suite')
    args = parse(command)
    assert args.epochs == args.schedule_epochs == 160
    assert args.stop_after_epoch == 130
    assert args.train_split == 'train+val+test' and args.eval_splits == ''
    assert args.lr == 1e-4 and args.backbone_lr == 3e-4
    assert args.warmup_epochs == 1 and args.seed == 0
    assert args.roi_context == 0 and args.roi_context_mode == 'shared'
    assert args.batch_size == 16 and args.workers == 2
    assert args.save_training_state and args.reproducible_loader
    assert args.resume is None and args.limit is None
    assert args.save_every == 10


def test_resume_preserves_real_training_compatibility_signature(tmp_path):
    import torch
    from sarbench.training_state import compatibility_signature, validate_compatibility

    data = tmp_path / 'data'
    for split in ('train', 'val', 'test'):
        for rel in (f'classification/{split}.csv', f'detection/instances_{split}.json'):
            path = data / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic manifest, same for save and resume')

    class Loader:
        dataset = range(9392)
        sampler = object()

        def __len__(self):
            return 587

    command = full.training_command('/python', data, tmp_path / 'suite')
    original = list(command)
    checkpoint = tmp_path / 'suite/full_train/training_state.pt'
    resumed = full.resume_command(command, checkpoint)
    assert command == original
    assert '--stop-after-epoch' not in resumed
    args = parse(resumed)
    assert args.epochs == args.schedule_epochs == 160 and args.resume == checkpoint
    assert args.stop_after_epoch is None
    model = torch.nn.Linear(2, 1)
    saved = compatibility_signature(parse(command), model, Loader(), 2)
    current = compatibility_signature(args, model, Loader(), 2)
    validate_compatibility(saved, current)
    args.schedule_epochs = 130
    with pytest.raises(ValueError, match='schedule_epochs'):
        validate_compatibility(saved, compatibility_signature(args, model, Loader(), 2))


@pytest.fixture
def submission(tmp_path):
    images = tmp_path / 'images.json'
    images.write_text(json.dumps({'images': [
        {'id': 1, 'width': 64, 'height': 32},
        {'id': 2, 'width': 64, 'height': 32},
    ]}))
    rows = ['1,0', '2,1']
    detections = [dict(image_id=i, category_id=i, bbox=[1, 2, 3, 4], score=0.8) for i in (1, 2)]

    def write(rows_=None, dets_=None):
        output = tmp_path / 'submission.zip'
        with zipfile.ZipFile(output, 'w') as archive:
            archive.writestr('classification.csv', 'image_id,label\n' + '\n'.join(rows if rows_ is None else rows_) + '\n')
            archive.writestr('detection.json', json.dumps(detections if dets_ is None else dets_))
        return output, images

    return write, rows, detections


def test_valid_submission_reports_coverage_without_upload(submission):
    write, _, _ = submission
    result = full.validate_zip(*write())
    assert result['prediction_images'] == 2 and result['boxes'] == 2
    assert result['format_passed'] and result['boxes_within_images']
    assert result['uploaded'] is False and len(result['sha256']) == 64


@pytest.mark.parametrize('fault', ['missing_classification', 'duplicate_classification',
                                   'missing_detection', 'nonfinite_box', 'nonfinite_score',
                                   'outside_box', 'too_many_boxes'])
def test_invalid_submission_is_rejected(submission, fault):
    write, rows, detections = submission
    rows, detections = list(rows), copy.deepcopy(detections)
    if fault == 'missing_classification':
        rows.pop()
    elif fault == 'duplicate_classification':
        rows[1] = rows[0]
    elif fault == 'missing_detection':
        detections.pop()
    elif fault == 'nonfinite_box':
        detections[0]['bbox'][0] = float('nan')
    elif fault == 'nonfinite_score':
        detections[0]['score'] = float('inf')
    elif fault == 'outside_box':
        detections[0]['bbox'] = [63, 2, 3, 4]
    elif fault == 'too_many_boxes':
        detections += [copy.deepcopy(detections[0]) for _ in range(100)]
    with pytest.raises((ValueError, AssertionError)):
        full.validate_zip(*write(rows, detections))
