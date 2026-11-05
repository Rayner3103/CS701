# BP7 自动选择时长并全数据训练

[English](README.md) · **中文**

2026-10-31 用户授权：“我睡了；你完成160轮之后，自行完成full train”。

队列 `/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full_selected_20261031`，控制器 `tools/run_bp7_full_selected.py`，启动PID3669143。等待 `BP7_fixedLR160_local_s0_20261030_direct` 完成最终验证并释放锁后自动执行，任何失败停止并保留现场。

## 固定决策

用统一submission后处理的本地val最终Δm比较原学习率130轮与160轮。160轮高出超过1.0则选择160，否则130。该门槛是操作性门槛而非统计显著性。按用户授权直接进入全数据训练，跳过第二seed复验，不声称稳定收益。决策写入selection.json。

## 全数据训练

- 原峰值LR固定：Head1e-4、Backbone3e-4。
- 选择的轮数同时作为余弦周期，warmup1轮，seed0。
- BP7架构不变，从通用预训练权重重新开始，未使用本地已训模型续训。
- 本地train+val+test合计9392，互不重叠，核验与官方训练集一致。官方val1426仅用于预测。
- 双卡、batch16、CPU0–3、nice10、每卡2 workers/threads。
- 每10轮保存推理权重，每轮保存完整断点。全量训练不再评估已纳入训练的本地val/test。
- 只执行一组全量训练，无无限搜索、无自动官方上传。

## 完成产物

自动用最终model.pt生成官方val submission，统一后处理及shift−0.2。检查格式、1426图片覆盖、框范围、每图最多100框、ZIP完整性及小于25MB，写入submission_validation.json和SHA256。ZIP位于仓库submissions，留给用户手动提交。

核验：130/160两种全量命令解析通过；学习率固定且无resume；9392数据并集核验通过；19项既有调度/来源校验测试通过。


## 2026-10-31 用户取消此队列

用户要求停止当前实验并直接启动分类上下文候选。全量队列在waiting阶段已停止，未执行全数据训练。当前实验见[BP7上下文160轮](../bp7_context160/README.zh-CN.md)。
