# SC2 backbone 候选与课程合规核查

核查日期：2026-10-31（新加坡时间）。对象：`51.86-full150ep-e130-DINOv3-LoRA80-Head1536-Cascade-NWD-SizeIoU-FlipBias-shift0.2`。

**结论：有来源和参数预算符合公开规则的替代方案，但没有找到同配置SAR实验证明必然胜过SC2的现成升级。** 真正较新的同规模候选是LingBot-Vision-B/16（2026-07）和TIPSv2-B/14（2026-04）；RADIOv3-B的timm固定512版本、DINOv3 ConvNeXt-B分别适合多教师特征和分层结构对照。

本轮为公开资料研究与CPU/meta参数审计，没有下载大权重、运行图像、训练、使用GPU或提交。结构计数不能代替权重加载、前向/梯度与准确率验证。既有训练代码和队列未改变。

## 官方规则

从用户指定网站核实[官方Terms](https://web-production-c241e.up.railway.app/docs#terms)及其明确链接的[公开文档资源](https://web-production-c241e.up.railway.app/assets/Docs-DK1tB6x4.js)。匿名访问`/team`只取得公共网页壳，没有查看团队私有账户内容。

- 通用预训练允许，示例包括ImageNet、CLIP、TerraMind和其他EO基础模型，须报告具体来源。
- 禁止额外SAR数据及在其上训练的检测/分类模型；规则明确区分一般卫星档案预训练的基础模型，不能将遥感或Sentinel-1预训练一律排除。
- 最终预测模型全部权重≤130M，冻结也计入；训练中更新的全部权重≤40M，包含heads。先全量微调再冻结不能回避统计。
- 单模型、每图一次输入、尺寸≤512×512；禁止ensemble、TTA、切图及超尺寸放大。官方val/test只供预测。
- 报告须披露模型来源、训练时教师、参数和全部调参计算成本、本地划分、seeds、消融；小于约2 Δm的提升需多seed均值和标准差支撑。

公开条款未逐项定义buffer。本地SC2检查额外计入buffer更保守；学到的权重不能因注册为buffer而免计。主候选已核算的buffer不影响预算通过。

## 参数核算

使用PyTorch meta设备构建层与张量形状求`numel`，不分配实际权重。SC2重建总数与备份`run/metrics.json`完全一致。SC2的原始灰度backbone为85,248,000，LoRA80为11,796,480；head/neck/RPN合计27,821,792，包含三个宽1536的Cascade阶段。

| 具体方案 | 裸backbone | 含head总参数 | 可训练参数 | 核算说明 |
|---|---:|---:|---:|---|
| SC2 DINOv3-B，灰度，r80 | 85,248,000 | 124,866,272 | 39,656,672 | 原快照重建，另16 buffer元素 |
| LingBot-B，灰度，r80 | 85,276,416 | 124,894,688 | 39,656,672 | 官方构造器+原head，另27,664 buffer元素 |
| LingBot-B，RGB，r80 | 85,669,632 | 125,287,904 | 39,656,672 | 保留原RGB patch结构 |
| TIPSv2-B/14，RGB，r80 | 86,316,288 | 125,934,560* | 39,656,672* | 只用视觉塔，32×32位置表 |
| RADIOv3-B，timm固定512，RGB，r80 | 86,456,064 | 126,074,336* | 39,655,136* | 官方转换架构 |
| DINOv3 ConvNeXt-B，灰度，MLP r80 | 87,560,320 | 127,116,192 | 39,594,016 | 换现有分层FPN，保留Cascade1536 |
| DINOv3 ConvNeXt-B，灰度，MLP r64 | 87,560,320 | 124,228,512 | 36,706,336 | 同上，adapter容量较小 |
| ConvNeXtV2-B，灰度，MLP r80 | 87,686,656 | 127,242,528 | 39,594,016 | 同上，GRN保持冻结 |

`*`：已计backbone/adapter，再加同参数量的SC2 heads，尚未完成端到端接入。TIPS的patch14坐标适配可能改变最终neck设计，不能把此数称为已验证可运行模型。所有方案均保留未合并的LoRA参数。RGB方案可从同一张灰度图构造三通道，仍只前向一次；预训练归一化需要随checkpoint处理。

相邻JSON分别保存[原模型与LingBot计数](vit_counts.json)、[TIPS计数](tips_counts.json)、[RADIO计数](radio_counts.json)、[ConvNeXt计数](convnext_counts.json)。

## 主候选

### LingBot-Vision-B/16：优先做同结构对照

[官方模型及权重](https://github.com/Robbyant/lingbot-vision#-model-zoo)描述通用网页图像、ImageNet、地标、街景等组织的约161M图像预训练，B由其通用Giant教师蒸馏。该来源类别符合课程通用预训练规则。

[论文Table 5](https://arxiv.org/html/2607.05247v1#S5.T5)同协议B级比较：DINOv3→LingBot的ImageNet linear为84.79→85.05、NYU RMSE为0.371→0.339、ADE mIoU为51.74→51.44、VOC为88.20→87.10。更新且深度更强，分割略弱；没有当前SAR检测结果。

768维、patch16、12层最接近现有结构。仍需适配字典输出、灰度输入、梯度检查点与masked bias。官方attention读取`qkv.in_features`，现有LoRA包装器未透传该属性，因此仅替换名称不能直接运行。审计的是结构参数，不是已接好的模型。

### TIPSv2-B/14：新增的2026年候选

[Google官方仓库](https://github.com/google-deepmind/tips)、[Base权重](https://huggingface.co/google/tipsv2-b14)、[论文](https://arxiv.org/html/2604.12012v1)。2026-04-13发布，通用WebLI图文数据和合成描述，约116M图像；只采用视觉塔符合通用预训练类别。原视觉+文字整套约196M，不能整体装入最终预测模型。

官方README写85.7M，但v2推理代码明确32×32位置表，本次RGB构造计数86,316,288，与论文约86.3M一致。论文“对DINOv3六项赢四项”是L级且包含图文任务，其ADE仍为51.4对54.9，不能推断B级SAR更强。

保留patch14可采用504×504单次输入，但要修正FPN、anchor、RoI尺度关系，现有stride16默认实现不能照搬。448也合规，但缩小目标。文本对齐的优势是否能改善固定类别SAR，需本地消融。

### C-RADIOv3-B：固定具体checkpoint后可用

[NVIDIA原装](https://huggingface.co/nvidia/C-RADIOv3-B)的128×128位置表占12,582,912参数，整个原装模块98,254,848，不能按表内约90M直接替换。原装RGB+r80+原head为137,873,120，超预算；r24可降至129,615,584/31,397,600，但adapter容量改变。

[timm固定512模型卡](https://huggingface.co/timm/vit_base_patch16_radio_v3.c_radio)明确：位置表以原方法一次重采样成32×32，无额外训练；512输出匹配原版，其他分辨率可能偏离。源权重revision为`44653a0482cf460bb4f12595fc3cc3dfecc403d1`。本次核实了官方说明与代码，未独立加载权重验证数值等价。

固定512版本86,456,064参数，保留RGB+r80+原head预算为126.07M/39.66M。数据为NV-CC-Img-Text，教师为DFN CLIP、SigLIP2、DINOv2、SAM，符合通用预训练类别；最终推理仅用学生视觉塔，不运行这些教师。v3预训练发布于2025；新timm移植不等于新一代预训练，也没有同配置SAR胜出证据。

### DINOv3 ConvNeXt-B：结构对照

[官方模型卡](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)说明采用通用LVD-1689M。它提供4/8/16/32分层特征；换用已有FPN、MLP两层LoRA80后，原Cascade1536可保留且预算通过。小尺度特征是实验动机，不是涨点证据：官方ADE的ConvNeXt-B为46.3，ViT-B为51.8。

旧项目训练过ConvNeXt-T且表现较弱，B/S当时只下载；不能把T的结果外推为B已失败。其方法与当前DINOv3同代。

## 其他候选与筛除理由

| 家族 | 结论 |
|---|---|
| [PE-Spatial-B16](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md) | 通用空间蒸馏，RGB约86.43M；2025-07早于DINOv3。裸Parameter attention投影需专门LoRA适配，未找到同配置SAR优势。 |
| [EVA-02-B/p16](https://github.com/baaivision/EVA/blob/master/EVA-02/det/README.md) | 官方已有patch14→16权重，约86M，通用ImageNet预训练。2023模型；COCO55.5AP使用1024输入及不同配方，不能直接迁移到此任务。 |
| [SAM2.1 Hiera-B+](https://github.com/facebookresearch/sam2#model-description) | 80.8M是完整SAM2而非裸骨干；可抽图像encoder做研究。默认1024 predictor不合尺寸要求，需≤512及移除不用的memory/prompt/mask组件；2024模型。 |
| [ConvNeXtV2](https://github.com/facebookresearch/ConvNeXt-V2)、[SwinV2](https://github.com/microsoft/Swin-Transformer)、[VMamba](https://github.com/MzeroMiko/VMamba)、[MambaVision](https://github.com/NVlabs/MambaVision)、[InternImage](https://github.com/OpenGVLab/InternImage) | 有通用预训练，可设计预算内方案；大多方法更早。算子、特征输出、适配器改动较多，原COCO成绩不等同512/40M适配预算的SAR比较。扩展结构计数见附加JSON。 |
| [Franca-B](https://github.com/valeoai/Franca) | 方法2025-07，通用IN21K；LingBot论文同协议B级表现不支持优先替换DINOv3。 |
| TerraMind、DOFA等通用EO模型 | 不能按遥感/SAR名称排除，TerraMind是条款明示允许的例子。当前8-bit高分灰度与卫星档案输入有域差；本项目TerraMind早期结果未证明更优。 |
| [DINOSAR](https://huggingface.co/YTang/DINOSAR) | 不纳入“确定合规”清单。[UniSAR-7M](https://huggingface.co/datasets/YTang/UniSAR-7M#data-sources-and-composition)直接包含SARDet-100K、ATRNet-STAR、FAIR-CSAR等训练图，部分使用标注筛选；不能直接套用一般卫星档案基础模型例外。 |
| DINOv3-L/SAT、RADIOv4-SO400M/H、PE大模型、SAM3原始encoder | 所涉公开型号单骨干已明显超预算，冻结/LoRA不改变最终权重总量。 |
| “DINOv4” | 本次未查到Meta官方可用发布，不列为可执行候选。 |

## 推荐实验顺序与解释边界

先比较DINOv3-B与LingBot-B，固定本地划分、head、日程和后处理，重点看mAP、AP75、small AP与弱类AP。第二组按问题选择：新的预训练方法用TIPSv2，多教师特征用RADIO固定512，多尺度结构用ConvNeXt-B。各方案独立训练，避免同时改变多个因素后把收益全部归于backbone。

本地holdout筛选；全量模型已见过的本地图不用于宣称泛化改善。微小提升需配对多seed复验。最终加载具体权重并完成接口后，重新确认所有推理权重、所有被训练权重、输入尺寸、单次前向及坐标一致性。本次没有新增准确率结果。
