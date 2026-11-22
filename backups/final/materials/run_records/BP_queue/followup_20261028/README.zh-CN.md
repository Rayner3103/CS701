# Bridge / playground 后续筛选

[English](README.md) · **中文**

准备于 2026-10-28，在 BP1/BP2/BP4 的结果出来之后。这些是本地消融，不是排行榜提交。正在运行的 BP5 → BP3 队列保持不变。

## 顺序和对照

1. 等待现有队列进程成功退出，并且 BP5 和 BP3 都有最终评估文件。
2. 在固定的、仅来自训练集的样本（每类最多 16 张图像）上诊断保存的 `AA_a4a2` 模型。记录正/负 RoI 数量、有和没有注入 GT 框时每个 GT 的覆盖情况、损失贡献以及局部 logit 梯度代理指标。不涉及优化器、反向传播或 checkpoint 修改。
3. 依次运行下面的筛选，每次完整运行之前先做一次简短的训练/checkpoint 加载烟测：

| 运行 | 相对 AA 的改动 | 主要对照 |
| --- | --- | --- |
| `BP7_spatial64` | `--spatial-prior 64` | AA：旧的 S5 空间分支在 A4 + flip-bias 下还有帮助吗？ |
| `BP8_ctx3_cls_only` | `--roi-context 3 --roi-context-mode classification` | BP1：扩大上下文的特征是否应只送入分类？ |
| `BP9_ctx3_clsw3` | `--roi-context 3 --cls-weights 3:3,7:3` | BP1/BP2：上下文和类别加权能否叠加？ |

所有筛选都使用现有的本地 train/val/test 划分（7,512/940/940）、seed 0、36 轮、LoRA r32/alpha64、head 宽度 1024、batch 16、AA 的 NWD/anchors/A4/flip-bias 配方和梯度检查点。这个队列中没有新的负样本损失规则、伪标签生成、外部数据或全数据训练。

`classification` 模式复用现有 head 权重：分类看到局部 + 扩大的 RoI 特征；框回归只看到局部 RoI。它增加 head 的计算量，但不增加参数。现有 checkpoint 默认为 `shared`。

## 资源和记录

- 只用 GPU0，这个队列中同一时间只有一个进程使用 GPU，并且在前一个进程退出之后才开始。骨干仍然对每张图像处理一次。
- Nice 级别 10；CPU 亲和性限制为四个 CPU；四个数据 worker 和库线程；最多两个编译器 worker。诊断使用两个线程和零个数据 worker。
- 源码复制到 `cs701_runs/BP_queue/followup_20261028/code/`，并附 SHA256 清单。每次训练运行再收到一份这个固定源码的副本。
- 运行状态、命令、诊断 JSON 和对照 JSON 放在 `cs701_runs/BP_queue/followup_20261028/` 下；各自的日志和 checkpoint 放在 `cs701_runs/BP7_spatial64/`、`BP8_ctx3_cls_only/`、`BP9_ctx3_clsw3/` 下。
- 前置条件、诊断、烟测、训练或评估失败时，队列停止并记录错误。现有实验目录绝不复用。

## 解读

用相同的设置比较最终后处理（包括本地 flip-bias 模型的 shift 0）。评估两个本地留出集、全部九个类别的 AP 和图像 macro-F1。图像类别乘法另外基于保存的原始预测进行评估。`comparison.json` 使用现有的本地参考 F1/mAP：val 91.50/22.41 和 test 91.61/23.31；这些不是官方排行榜参考值。

单个 seed 只用于筛选想法，不能证明可重复性。在改动损失归一化之前，先检查采样诊断。不要把附加 GT 后的覆盖率解读为 proposal 召回率，也不要把 logit 梯度代理指标解读为骨干的梯度冲突。在组合改动或启动放大/全数据运行之前，重新核对参数预算和验证证据。
