# SC2 e130：RPN 2000 候选的 Phase 1 submission

2026-10-30（SGT）。已生成并检查本地文件，尚未上传或获得官方分数。

提交：`submissions/SC2_e130_val_shift-0.2_rpn2000.zip`。

模型来自 51.86 备份的 `weights/model.pt`，加载同一冻结 DINOv3 预训练骨干及可训练权重。推理源码全部使用该备份 `code/`；当前工作区的其他训练修改未混入。

唯一模型配置变化：RPN testing 的 pre-NMS 和 post-NMS 上限均由 1000 提高到 2000。Training 上限仍为 2000，RPN NMS 阈值仍为0.7。输入512×512、bf16、分类融合λ1、Soft-NMS σ0.75、objectness^0.25、三阶段框平均、shift=-0.2、score阈值1e-5、min_keep30、最终每类/全图最多100框全部沿用原设置，无框投票。

一次前向同时产生分类和检测。分类仍使用同一次2000候选输出的检测辅助融合，不从旧提交复制分类结果；最终核对分类CSV恰好与原提交逐字节一致。

## 运行和检查

- GPU1、batch2、2个workers、2个CPU线程、nice10；PyTorch显存分配上限为设备总量的12%。未停止或修改正在进行的训练。
- 全部1426张图完成，共713个batch前向，每张图只输入一次；缓存确认每张图均有2000个进入Cascade的RPN候选。
- 参数仍为总124,866,272、可训练39,656,672。
- 端到端模型预测约53.2ms/图；包含加载和写文件的记录约78.23秒。GPU显存峰值约0.96GiB。与其他进程共享GPU，不视为独占性能测试。
- 相比原默认batch16，使用batch2控制资源；bf16批大小变化可能带来轻微数值舍入差异。
- 提交检查通过：1426张图、61,896个框、无缺框图、无越界框，每图每类最多100框。
- 原提交有61,683框，本次61,896框；分类标签变化0张。
- ZIP约9.56MB，SHA256：`06d9b37e934c43632aaf0067d8e517b5e911e248fd63d15cc1e13b238513defa`。

`manifest.json` 保存模型/源码/输入/输出哈希、完整配置、实际前向与候选计数；`status.json` 记录完成状态。原始输出保留于 `submissions/raw/SC2_e130_val_rpn2000_raw.pt`，便于后续CPU分析。

## 复现

从仓库根目录运行。先选择有余量的GPU，输出路径必须不存在：

```bash
CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/cs701bench/bin/python -u tools/make_rpn2000_submission.py \
  --out submissions/SC2_e130_rpn2000_reproduced.zip \
  --raw-out submissions/raw/SC2_e130_rpn2000_reproduced.pt \
  --record-dir experiments/sc2_rpn2000_reproduced
~/miniconda3/envs/cs701bench/bin/python tools/check_submission.py \
  submissions/SC2_e130_rpn2000_reproduced.zip \
  ~/cs701_data/cs701-sar-course-data/val/images.json
```
