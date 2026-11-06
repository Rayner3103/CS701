# Code map

| File | Role |
|---|---|
| code/sarbench/instance_sampling.py | Main method change: instance-balanced positive sampling |
| code/sarbench/data.py | Native image geometry, augmentation, metadata and coordinate inverse |
| code/sarbench/model.py | Joint model, valid-region pooling and padded-canvas RoI geometry |
| code/sarbench/tiny.py | Upstream F2 Cascade, matching, anchors and compact heads |
| code/sarbench/backbones.py | DINOv3 construction; local pretrained-file support |
| code/train.py | Reused upstream model builder/optimizer; activates new options; do not use original main |
| code/run_project.py | Actual training/validation/checkpoint runner |
| code/run_safety.py | Scoring, partition/source/data identity and resume checks |
| code/native_postprocessing.py | Fixed native inference postprocessing and class fusion |
| code/predict_native.py | Correct full-checkpoint inference/export entry |
| code/prepare_native_data.py | Rebuild recipient-local CSV/COCO metadata using fixed split |
| code/supervise.py | Optional supervised launcher; run directly for ordinary training |
| code/recipe.py | Historical experiment recipe with original machine paths |
| configs/train_arguments.json | Portable fresh-training argument array |
| configs/split_seed701.json | Exact internal split |
| results/ | Original experiment settings, selection and metrics |
| evidence/ | Completion and bounded qualification evidence |

The remaining Python files are retained upstream/operational helpers, not the recommended portable entry points. Historical qualification scripts contain local D-drive paths; their recorded evidence is included, but the quick-start commands are the client workflow.
