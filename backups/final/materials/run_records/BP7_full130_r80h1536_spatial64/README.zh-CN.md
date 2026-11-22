# BP7 全量大模型：130轮双卡训练

[English](README.md) · **中文**

本次按用户指令，将 SC2 大模型配置（LoRA rank 80 / alpha 160、Cascade head 1536）与 BP7 的 SpatialPrior(64) 结合。

- 从通用 DINOv3 预训练权重初始化，新增模块从头训练；不加载 BP7 小模型或 SC2 微调权重。
- 仅训练官方 train 的全部 9,392 张图。本地 train+val+test 是官方 train 的内部划分，已逐集合核对。
- 正式训练 130 轮，按130轮余弦学习率衰减；不是旧 SC2 的150轮计划中第130轮。
- GPU0、GPU1，DDP全局batch16，每卡8；总CPU限制4核，每rank2个数据worker。
- 总参数125,076,960，可训练39,867,360；梯度检查点、compile、A4、NWD8、flip-bias -0.3沿用SC2。
- 正式训练前做双卡3步smoke，验证保存权重加载和8张官方val预测格式。smoke权重不用于正式训练。
- 每10轮保存检查点，最终第130轮保存为model.pt。
- 完成后自动对官方Phase1 val生成shift -0.2预测ZIP与原始输出，只保存本地，不自动上传。

运行状态见status.json，调度日志driver.log，正式训练stdout.log/log.txt，代码快照code/，哈希清单source.sha256。启动参数见commands.json。
