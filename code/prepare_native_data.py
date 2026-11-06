"""Create F2's CSV/COCO layout without duplicating course images or changing the split."""
import argparse
import csv
import json
from pathlib import Path

from run_safety import atomic_json, sha256, validate_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-train', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    coco_path = args.course_train/'instances.json'
    coco = json.loads(coco_path.read_text(encoding='utf-8'))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    validate_partition(manifest, coco)
    assert len(manifest['train']) == 7984 and len(manifest['val']) == 1408
    image_map = {int(i['id']): i for i in coco['images']}
    if args.out.exists():
        raise FileExistsError('Use a new output directory')
    (args.out/'classification').mkdir(parents=True)
    (args.out/'detection').mkdir()
    for part in ('train','val'):
        ids = {int(r['image_id']) for r in manifest[part]}
        new_images = []
        with (args.out/'classification'/f'{part}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream); writer.writerow(['image','label'])
            for row in manifest[part]:
                image = dict(image_map[int(row['image_id'])])
                path = (args.course_train/image['file_name']).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                image['file_name'] = path.as_posix()
                writer.writerow([path.as_posix(), int(row['label'])]); new_images.append(image)
        atomic_json(args.out/'detection'/f'instances_{part}.json',
                    {**coco, 'images': new_images,
                     'annotations': [a for a in coco['annotations'] if int(a['image_id']) in ids]})
    atomic_json(args.out/'DATA_PROVENANCE.json', {'course_metadata_sha256': sha256(coco_path),
                'manifest_file_sha256': sha256(args.manifest), 'audited_partition_sha256': manifest['split_sha256'],
                'train_images': 7984, 'val_images': 1408,
                'images': 'Original files referenced by absolute path; no copies, tiling or upscaling.'})
    print(json.dumps({'status':'PREPARED','output':str(args.out),'train':7984,'val':1408}))


if __name__ == '__main__':
    main()
