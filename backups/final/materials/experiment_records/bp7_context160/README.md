# BP7 + 仅分类上下文：160轮本地对照

2026-10-31用户明确要求停止当前实验，直接启动autoresearch第一候选。已停止原160轮训练（e154中途，保留e153完整断点）及尚未启动的全量队列，没有删除原结果。

## 设置

- 相同BP7源码，架构配置变化为 `--roi-context 3 --roi-context-mode classification`。
- 检测RoI分类分支使用扩大上下文；框回归及Cascade阶段间框更新仍使用局部RoI。共享head权重。图像分类head没有直接修改。
- Head峰值LR1e-4、Backbone3e-4，160轮训练/余弦，warmup1轮，seed0，train7512/val940。
- 从通用预训练权重开始，不加载已训BP7权重。上下文模块会改变后续初始化随机数消耗，不能声称未改模块初值逐位相同。
- 双卡、batch16、CPU0–3、nice10、每rank2 workers/threads。
- 三个ctx_reduce共增加49,344参数：总125,126,304，可训练39,916,704，余83,296训练参数。双卡2步smoke检查实际参数与保存状态，再开始正式训练。

运行目录：`/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_ctx3_classification_local160_s0_20261031`。控制器：`tools/run_bp7_context_trial.py`，初始PID3745643。

## 对照

原160轮被中止，不能把新e160对比旧e130称为单因素消融。主要对照固定为新context e150与原160-cos方案保存的e150，数据划分、日程和seed一致，统一最终submission后处理及shift−0.2。新e160单独评估。

本队列不自动重启旧训练、不自动全数据重训、不上传官方提交。结果保存在baseline_e150_post_val.json、context_e150_post_val.json、context_e160_post_val.json及comparison.json。

## 检查

既有19项ROI路由/梯度/旧权重兼容测试及新增19项控制器参数、预算、结果来源校验均通过。新命令与原160方案仅增加上下文两flag。

manifest的expected_parameters/expected_trainable_parameters用于原实现一致性检查；context_expected_*为新模型预算。每10轮保存推理权重，每轮保存完整状态，失败停止，不盲目重试。


## 2026-10-31 完成结果

训练及评估于11:46完成，训练耗时 7.930 小时。以下全部为统一submission后处理的本地val，不能直接与官方Δm比较。

| 模型 | mAP | Macro-F1 | AP50 | 本地Δm |
|---|---:|---:|---:|---:|
|原160-cos e150|52.6324%|95.5099%|73.1851%|69.6011|
|上下文160-cos e150|53.5982%|95.6948%|73.9320%|71.8566|
|上下文最终e160|53.6858%|95.6026%|73.9678%|72.0018|

同e150匹配对照本地Δm提升 2.2555，mAP约+0.966个百分点；单seed结论。用户另要求生成最终e160官方val submission，使用最终model.pt及shift−0.2，尚未上传。


## 官方反馈：e160 submission

2026-10-31 11:50，用户截图：Δm46.80、Macro-F1 94.89%、mAP50.15%、accuracy95.16%、AP50 72.02%。按截图显示精度保存，未反推更多小数。

相比无上下文的160-cos e130提交（47.12/94.58/50.41/94.95/71.62），Δm−0.32、mAP−0.26个百分点，F1+0.31、accuracy+0.21、AP50+0.40个百分点。该官方比较同时改变了检查点轮数，不能单独归因于上下文。局部e150配对优势未在当前e160官方提交上表现为领先，单seed小差异不能视为统计结论。相比原130-cos e130提交45.83，当前Δm+0.97，但也不是纯上下文对照。

评分与文件哈希已记录在运行目录context160/submission_e160/leaderboard_score.json。尚未据此启动新的训练或提交。
