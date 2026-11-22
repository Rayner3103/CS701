# 零训练拼接 submission bbb.zip

[English](README.md) · **中文**

用户要求完全不训练地换LingBot-B并生成submission。为与aaa.zip对照，固定复用同一BP7 full e50的分类头、neck、RPN、Cascade，共88个张量、28032480参数；未复制任何DINOv3 backbone或adapter张量。

LingBot使用已固定并核验SHA256的官方Base预训练。新LoRA up为零，骨干所有原始张量逐项核对与官方权重一致。拼接后没有创建optimizer或执行训练更新，推理时所有参数冻结，eval/no_grad。输入、官方val1426、batch2及后处理与aaa一致，显式shift−0.2。

已生成submissions/bbb.zip与bbb.sha256，通过格式、图片覆盖、有限值、框范围、每图100框、ZIP完整性及25MB限制校验。未上传。来源与模型校验详见assembly.json，生成命令见launch.json，提交校验见validation.json。

model/log.txt首行仅沿用重建架构所需字段及head供体的训练配方，不代表拼接后的模型又训练了这些轮数。拼接后新增训练步数为0；供体head已经在官方train9392上训练50轮。prediction_comparison.json仅比较无标签预测，不能当作准确率或泛化指标。
