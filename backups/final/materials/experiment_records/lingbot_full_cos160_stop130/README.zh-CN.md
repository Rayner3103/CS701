# LingBot-Vision-Base 替换 BP7 骨干

[English](README.md) · **中文**

2026-10-31 用户授权：“这个结束后，换LingBot-B试试”。当前BP7完成e130并导出submission后，自动执行一组LingBot-B对照。先做短双卡烟测，通过才正式训练；失败保留现场并停止。

## 固定配置

- LingBot-Vision-Base ViT-B/16、768维、12层，替换DINOv3-B。
- 保留原BP7 LoRA80/alpha160、Cascade1536、SpatialPrior64、NWD8、SizeIoU、aligned RoI、flip-bias−0.3及anchors；无context。
- 官方train全量9392张。本地train7512+val940+test940实际路径均核验指向官方train；官方val1426只推理。
- Head LR1e-4、Backbone LR3e-4、warmup1轮、weight decay0.05、seed0、global batch16。
- 160轮余弦，在e130停止，76310更新步骤。每轮保存完整断点，每10轮保存推理权重，不自动继续至160。
- 双卡，CPU0–3，nice10，每卡2个worker/计算线程。
- 灰度patch kernel由RGB权重求和，保持相同SAR归一化。关闭LingBot预训练的随机RoPE坐标缩放，保留RoPE位置编码，与现有BP7下游增强配方一致。
- e130自动导出 submissions/LingBotB_full_cos160_e130.zip，历史后处理及shift−0.2；单图单次512前向。校验1426图覆盖、格式、有限框、边界、每图100框、ZIP完整性及25MB限制，保存SHA256。用户手动提交。

## 固定来源及验证

官方代码 https://github.com/Robbyant/lingbot-vision 固定commit 151e46321bae4399f8568829f190c7bdec216b49。原文件及LICENSE/LEGAL/NOTICE/UPSTREAM保存在sarbench/vendor/lingbot_vision，适配独立位于sarbench/lingbot.py。

官方Base权重 https://huggingface.co/robbyant/lingbot-vision-vit-base 固定revision f606f8c6c4002234ea68038f4d7c7cf57da96dfa。model.pt为342852282字节，SHA256 783dfb59014c34e9f0013db60bf6f5cc3c6b0604eb9528fb5df65e80769c5825。下载核验见pretrained.json。

公开来源为通用图像自监督预训练，Base由通用Giant蒸馏，仅使用Base视觉骨干。按公开课程条款 https://web-production-c241e.up.railway.app/docs#terms 允许通用预训练。实测总参数125105376、可训练39867360、浮点buffers27664，参数加buffers125133040，符合130M/40M预算。

真实权重CPU检查：严格加载、灰度求和与RGB重复通道前向误差小于5e-6、48层LoRA有限梯度、gradient checkpoint前向/梯度一致、Dynamo单block fullgraph+checkpoint通过。CPU不能代替CUDA/Inductor双卡烟测，后者排在当前BP7完成之后。

## 队列与解释

前序：/mnt/hdd/junkaichen/CS701/cs701_runs/BP7_full9392_cos160_stop130_s0_20261031。

新队列：/mnt/hdd/junkaichen/CS701/cs701_runs/LingBotB_full9392_cos160_stop130_s0_20261031。控制器tools/run_lingbot_full_stop130.py，等待前序e130权重及submission校验通过并释放锁。18项队列CPU测试通过。

当前BP7冻结快照不修改。尚无本SAR任务的效果证据。全量训练后的本地val/test已参与训练，不用于宣称泛化改善；与相同配方BP7的官方e130结果比较。单seed小差异不能认定稳定提升。完整记录烟测、正式训练及搜索成本。

等待队列已启动，PID2066031，状态waiting_for_bp7；合计42项相关CPU测试通过。GPU烟测在前序完成后执行。实时状态以运行目录status.json为准。

## 2026-10-31 19:07 SGT 运行更新

BP7 已于18:44完成e130并导出。原LingBot队列18:45在双卡烟测的第一次反向传播失败：逐block torch.compile与外层非重入gradient checkpoint组合触发重算张量元数据不一致检查；正式训练未启动。原失败目录、源码和日志完整保留。

仅移除LingBot启动命令的`--compile`，保留梯度检查点及一致性检查、bf16、batch16、原学习率与训练日程。启动配置和断点/提交校验28项CPU测试通过。恢复脚本`recover_nocompile.py`从原冻结快照复制，现有源码仅替换启动器及其回归测试；训练脚本、骨干和head代码未变。

新运行目录：`/mnt/hdd/junkaichen/CS701/cs701_runs/LingBotB_full9392_cos160_stop130_s0_20261031_nocompile`。控制器PID1432089；启动记录`recovery_launch.json`。双卡48图、3步训练、完整断点及8图预测提交校验全部通过，19:07:08正式启动9392图全量训练，仍在e130停止并导出同名LingBot submission。实时状态以新目录的`status.json`和`full_train/log.txt`为准。
