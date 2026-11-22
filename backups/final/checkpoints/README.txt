val/test 最终推理所需 checkpoint 备份

GitHub Release（两份权重，共约 502 MB）：
https://github.com/Rayner3103/CS701/releases/tag/final-val-test-checkpoints-20261122

配置、来源、校验清单和本脚本保存在仓库 junkai 分支。
从 GitHub clone/pull 后，在仓库根目录运行：
  python3 backups/final/checkpoints/download.py
脚本会将两个 Release 附件还原到本目录的正确路径，并验证 SHA256。
仅检查已有文件：
  python3 backups/final/checkpoints/download.py --check

两个阶段共用 avg_e100-180.pt（epoch 100,110,...,180 的参数平均）。
该文件仅包含 234 个可训练张量，因此同时备份了必要的 DINOv3 冻结骨干
预训练权重（hf_cache/）以及构建模型需要的 metrics.json。
文件均为独立复制，已逐文件核对 SHA256，没有指向外部的软链接或硬链接。
九个源 epoch checkpoint 和 optimizer/training_state 不参与最终推理，未重复复制。
本目录位于材料重建目录 materials/ 之外，build_materials.sh 重建时会保留。

校验：在本目录运行 sha256sum -c SHA256SUMS。
manifest.json 记录文件来源、哈希、共享权重和两个阶段的后处理设置。
原始输出缓存和提交文件已存在 ../materials/local_only/，不是模型 checkpoint。

重新推理（仓库根目录，使用现有 cs701bench 环境）：
  export HF_HUB_CACHE="$PWD/backups/final/checkpoints/hf_cache"
  export HF_HUB_OFFLINE=1
  python predict.py --run backups/final/checkpoints --weights avg_e100-180.pt \
    --images /path/to/course/val --batch-size 16 --workers 2 \
    --rpn-topn 2000 --obj-pow 0 --soft-sigma 0.5 --stage-box mean \
    --cls-lambda 2 --shift -0.2 --fusion agn --gamma 1 --topk 9 \
    --out /path/to/val.zip

val 56.49 使用上述设置。test 普通候选使用同一权重和后处理，历史 batch-size 为 2。
生成 test FINAL 候选：将 images 改为 test、batch-size 改为 2，增加
--min-keep 100，并输出 /path/to/test_mk100.zip；随后运行：
  python tools/snap_submission.py /path/to/test_mk100.zip \
    /path/to/course/test/images.json /path/to/test_FINAL.zip
  python tools/check_submission.py /path/to/test_FINAL.zip /path/to/course/test/images.json

报告记录 test 分数 61.25，但无法确认当时手动上传的是普通候选还是 FINAL 候选。
两者均使用这里备份的同一份平均 checkpoint。不同 batch/GPU 的浮点差异可能
使少量检测框发生变化；要逐字节重建历史提交内容，应使用已有 raw 缓存。
