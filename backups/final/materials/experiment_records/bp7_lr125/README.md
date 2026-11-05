# BP7：学习率提高 25%，130 轮

2026-10-30 用户确认优先试提高学习率，暂缓原定 160 轮实验。

| 设置 | 原学习率基线 | 新对照 lr125x |
|---|---:|---:|
| Head LR | 1e-4 | 1.25e-4 |
| Backbone LR | 3e-4 | 3.75e-4 |
| Epoch / cosine schedule | 130 / 130 | 130 / 130 |
| Warmup | 1 epoch | 1 epoch |
| Seed | 0 | 0 |
| 训练 / 本地验证 | 7512 / 940 | 7512 / 940 |

固定 BP7 架构、初始化方式、数据增强、优化器、全局 batch16、双卡、CPU0–3、nice10、每卡2 workers/threads。新组从通用预训练权重初始化，不续训已有模型。学习率之外的命令参数与基线一致。

## 队列与交接

- 新目录：`/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_LR125_local_s0_20261030`。
- 代码：`tools/run_bp7_lr125_trial.py`，启动使用新目录中的固定源码快照。
- 18:04 启动新控制器 PID2665530，已进入 `waiting_for_hybrid130`。
- 只终止旧控制器 PID1528519，保留 torchrun553814 和训练 rank554080/554081；旧目录状态标记 `superseded`，保留原计划和全部产物。
- 等 hybrid130 正常结束，确认完整130轮训练状态并完成最终统一后处理验证，然后自动启动 lr125x。
- 原 `schedule160` 不会自动启动。新队列只启动这一组新训练，完成后停止，无自动上传、全数据重训或本地 test 评估。

## 比较与检查

主比较为最终 epoch130、相同 submission 后处理的本地 val 指标，与 `BP7_LR130_local_s0_20261029_v2/lr1x` 比较。报告 mAP、Macro-F1、accuracy、AP50 和本地 Δm 差值，不把单 seed 收益当成稳定结论。

`status.json` 记录状态，`handoff.json` 记录旧进程身份和交接前状态，`commands.json` 记录实际执行命令，`comparison.json` 保存最终对比。失败停止并保留日志，禁止并行重叠启动。

检查通过：19 项已有策略/来源校验测试；两组命令只有两个 LR 数值改变；进程身份与退出检测；实际交接后旧控制器退出、原双卡训练继续。
