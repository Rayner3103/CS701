"""Exploratory follow-up: apply the fixed .8/.10 vote only to small predictions.

Use the original, pre-vote sqrt(area) in 512-input coordinates as the gate.
Compare <16 and <32 on local val. Confirm only the val winner on local test,
and only if it improves total mAP. This follows observed val size diagnostics;
it is an exploratory screen, not an independent hypothesis test.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_box_voting import (ROOT, evaluate_predictions, make_predictions,
                                      sha256, validate_holdouts)


OUT = ROOT / 'experiments/box_voting_sc1'
DATA = Path.home() / 'cs701_data/dataset/SARFact-Course-20K'
RUN = Path.home() / 'cs701_runs/SC1_r80h1536'
VOTE = dict(name='vote_iou0.8_alpha0.10', iou_threshold=.8, alpha=.1)


def gated_predictions(base_path, voted_path, split, side_cutoff):
    base = json.loads(base_path.read_text())
    voted = json.loads(voted_path.read_text())
    assert len(base) == len(voted)
    images = {im['id']: im for im in json.loads(
        (DATA / 'detection' / f'instances_{split}.json').read_text())['images']}
    mixed, eligible = [], 0
    for original, altered in zip(base, voted):
        assert {k: v for k, v in original.items() if k != 'bbox'} == {
            k: v for k, v in altered.items() if k != 'bbox'}
        im = images[original['image_id']]
        side = math.sqrt(original['bbox'][2] * original['bbox'][3]
                         * 512 ** 2 / (im['width'] * im['height']))
        small = side < side_cutoff
        eligible += small
        mixed.append(altered if small else original)
    path = OUT / f'{split}_vote_lt{side_cutoff}.json'
    path.write_text(json.dumps(mixed, separators=(',', ':')))
    return path, eligible


def main():
    torch.set_num_threads(1)
    validate_holdouts(json.loads((RUN / 'metrics.json').read_text())['args'], DATA, ['val', 'test'])
    previous = json.loads((OUT / 'results.json').read_text())
    base_path = OUT / 'val_baseline.json'
    vote_path = OUT / 'val_vote_iou0.8_alpha0.10.json'
    paths, counts = {}, {}
    for cutoff in (16, 32):
        paths[f'lt{cutoff}'], counts[f'lt{cutoff}'] = gated_predictions(base_path, vote_path, 'val', cutoff)
    metrics = {'baseline': previous['val']['metrics']['baseline'], **evaluate_predictions(
        paths, DATA / 'detection/instances_val.json', 2)}
    selected = max(metrics, key=lambda name: metrics[name]['mAP'])
    result = dict(exploratory=True, fixed_vote=VOTE, cutoffs=[16, 32],
                  gate='pre-vote predicted sqrt(area), 512-input pixels; strict < cutoff',
                  script_sha256=sha256(Path(__file__)),
                  input_sha256={p.name: sha256(p) for p in (base_path, vote_path)},
                  val=dict(metrics=metrics, eligible_boxes=counts), selected_on_val=selected)
    result_path = OUT / 'size_gate_results.json'
    result_path.write_text(json.dumps(result, indent=2))
    print(f'Frozen validation choice: {selected}', flush=True)
    if selected != 'baseline':
        cutoff = int(selected.removeprefix('lt'))
        paths, metadata = make_predictions(RUN, DATA, 'test', [VOTE], OUT)
        chosen_path, count = gated_predictions(paths['baseline'], paths[VOTE['name']], 'test', cutoff)
        test = evaluate_predictions({'baseline': paths['baseline'], selected: chosen_path},
                                    DATA / 'detection/instances_test.json', 2)
        result['test'] = dict(metadata=metadata, metrics=test, eligible_boxes=count)
        result['improves_both_holdouts'] = test[selected]['mAP'] > test['baseline']['mAP']
    else:
        result['test'] = {'skipped': 'No total val mAP improvement'}
        result['improves_both_holdouts'] = False
    result_path.write_text(json.dumps(result, indent=2))
    print(f'Results saved to {result_path}', flush=True)


if __name__ == '__main__':
    main()
