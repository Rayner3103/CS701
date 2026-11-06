import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_run_helpers_exist():
    assert importlib.util.find_spec('run_safety') is not None


def test_official_formula_internal_scope_and_invalid_metrics():
    safety = importlib.import_module('run_safety')
    assert safety.internal_delta(.9331, .2613) == pytest.approx(0)
    assert safety.internal_delta(.9331, .5226) == pytest.approx(50)
    for value in (float('nan'), -1, 1.01):
        with pytest.raises(ValueError):
            safety.internal_delta(value, .2)


def test_resume_refuses_sampler_or_source_change():
    safety = importlib.import_module('run_safety')
    original = {'epochs': 100, 'instance_balanced': True, 'data_fingerprint': 'a',
                'source_hashes': {'model.py': 'b'}, 'batch_size': 2, 'accumulation': 8}
    safety.check_resume(original, dict(original))
    for key, value in [('instance_balanced', False), ('epochs', 36),
                       ('data_fingerprint', 'c'), ('source_hashes', {'model.py': 'd'})]:
        with pytest.raises(ValueError, match=key):
            safety.check_resume(original, {**original, key: value})


def test_accumulation_tail_uses_actual_group_length():
    safety = importlib.import_module('run_safety')
    assert [safety.group_length(i, 10, 4) for i in range(10)] == [4]*8 + [2]*2


def test_partition_rejects_overlap_and_wrong_class():
    safety = importlib.import_module('run_safety')
    manifest = {'train': [{'image_id': 1, 'label': 0}], 'val': [{'image_id': 2, 'label': 1}]}
    coco = {'images': [{'id': 1}, {'id': 2}],
            'annotations': [{'image_id': 1, 'category_id': 1}, {'image_id': 2, 'category_id': 2}]}
    safety.validate_partition(manifest, coco)
    with pytest.raises(ValueError, match='overlap'):
        safety.validate_partition({**manifest, 'val': manifest['train']}, coco)
    with pytest.raises(ValueError, match='class'):
        safety.validate_partition(manifest, {**coco, 'annotations': [{'image_id': 1, 'category_id': 2}]})
