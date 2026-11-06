# Quick start

## Environment

Tested: Windows, Python3.11.4, NVIDIA RTX3060 Laptop6GB, CUDA11.8 PyTorch wheels. Training requires CUDA BF16; the supplied training/inference entry points are not CPU or Mac-MPS implementations. The original upstream Blackwell/TerraTorch requirements are archived in `UPSTREAM_REQUIREMENTS.txt`; use the delivery-root requirements for this DINO-only experiment.

Run from the extracted delivery root:

```text
python -m venv .venv
```

Activate on Windows PowerShell with `.venv/Scripts/Activate.ps1`, or use the environment's Python explicitly. On Linux use `source .venv/bin/activate`.

```text
python -m pip install -r requirements.txt
```

## Prepare the exact partition

Place the official labeled training files under `data/course_train/`, containing `instances.json` and the original image files referenced by its `images[].file_name`. The raw images are not bundled. Regenerate prepared CSV/COCO files for the recipient's machine; original experiment CSV files contain machine-specific paths and are deliberately not supplied as runnable data.

```text
python code/prepare_native_data.py --course-train data/course_train --manifest configs/split_seed701.json --out data/prepared
```

The split is 7,984 train /1,408 validation, seed701. Classification labels are0..8, detection categories1..9. The script checks image coverage, class consistency and partition separation.

## Generic pretrained initialization for fresh training

Download `model.safetensors` from the pinned timm repository revision below and save as `weights/dinov3_b16.safetensors`:

https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/resolve/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/model.safetensors

SHA-256: `1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332`.

## Fresh training

```text
python code/run_project.py --data data/prepared --out outputs --name JIB_reproduction701 --epochs 100 --batch-size 4 --accumulation 4 --workers 0 --seed 701 --backbone dinov3 --init pretrained --adapt lora --pretrained-file weights/dinov3_b16.safetensors --lora-targets attn.qkv,attn.proj,mlp.fc1,mlp.fc2 --lora-rank 32 --lora-alpha 64 --train-norms --backbone-lr 0.0003 --lr 0.0001 --cls-weight 1.0 --weight-decay 0.05 --cascade --cascade-rep 1024 --cascade-beta 0 --cascade-size-iou --size-iou-table a4 --aligned-roi --rpn-nwd 8 --anchor-offset 0.5 --anchor-sizes "6,8,11;16,19,23;32,38,45;64,76,91;128,152,181" --flip-bias -0.3 --native-size --instance-balanced --train-split train --eval-splits val --eval-every 4
```

Use a new empty output run. The equivalent argument array is in `configs/train_arguments.json`. The original `code/recipe.py` retains historical machine paths and is not the portable launch command. Evaluate epoch1, multiples of4 and100; select maximum internal Delta_m over those26 evaluations. Other epochs were not validated.

For recovery of a NEW run on the same machine, repeat the exact command and add `--resume outputs/JIB_reproduction701/last_checkpoint.pt`. Resume accepts only the same run's latest complete-epoch file and unchanged source, configuration and data fingerprints. Do not resume from the delivered epoch88 selected checkpoint, or expect an original run to resume after relocating paths. The provided selected checkpoint is for inference.

## Prediction with the delivered selected model

Use the companion with-model ZIP, or copy its `weights/JIB_epoch88.pt` into the code-only package. No generic initialization download is needed for this inference command.

Put target images under `data/inference/`, with `images.json` containing an `images` array of objects with integer `id`, `file_name`, `width`, `height`. File names resolve relative to that folder. Example: `{"images":[{"id":1,"file_name":"images/00001.jpg","width":512,"height":512}]}`. Use actual image dimensions and IDs.

```text
python code/predict_native.py --checkpoint weights/JIB_epoch88.pt --images data/inference --out outputs/inference_epoch88 --batch-size 4
```

The output directory must not already exist. Outputs: `classification.csv`, `detection.json`, `submission.zip` and `INFERENCE_PROVENANCE.json`. The ZIP contains only the two prediction files, without uploading. Confirm the target evaluation system's current schema before submission. Model weights use PyTorch's full trusted-checkpoint format.

## Focused implementation checks

From the `code` directory:

```text
python -m pytest tests/test_instance_sampling.py tests/test_native_geometry.py tests/test_run_safety.py tests/test_standalone_runner.py tests/test_postprocessing.py -q
```

The delivered code is source-identical to the completed model implementation. Data paths and exact execution environment affect reproduction. One seed provides no stability estimate or causal proof of improvement.
