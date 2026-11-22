# CS701 autoresearch：规则、已有证据与下一轮候选

[English](README.md) · **中文**

研究时间：2026-10-31，新加坡时间。用户允许考虑符合规则的各种改进；本轮未改变训练队列、训练权重或提交设置，未启动额外GPU任务。完成了官方规则核查、历史实验审计、训练框统计以及已有预测上的CPU TIDE诊断。

## 结论

当前最有依据的方向是**减少弱类漏检、改善严格IoU下的定位**。优先验证BP7与仅分类上下文的组合，以及诊断驱动的正样本匹配/框回归；继续对峰值LR盲扫、简单增大RPN候选数或调NMS，证据更弱。新的设计先在本地训练划分测试，不以官方提交作为搜索循环。

## 规则边界（实时核查）

依据[官方Docs](https://web-production-c241e.up.railway.app/docs)及该页面2026-10-31公开组件Docs-DK1tB6x4.js：单模型、每图一次输入、尺寸不超过512×512；总参数≤130M、训练更新的参数≤40M。禁止预测集成、TTA、切片、额外SAR数据与在额外SAR检测/分类数据上训练的模型。官方val/test只可预测，不可伪标训练、适应或手工标注；本地940/940来自官方train，最终全量训练可以纳入。

通用预训练可以使用并须报告来源。训练-only教师被报告要求提及，但教师来源和训练参数统计仍须满足约束；不把这解读成任意教师都获准。EMA/SWA/权重平均未被单独裁定，暂列需确认，不自动生成平均权重提交。参数冻结后曾被更新过的权重仍需如实披露，不以某一步的可训练参数数目规避上限。

当前模型总参数125,076,960、训练参数39,867,360，仅剩132,640训练参数空间。

Phase1参考F1=93.31%、mAP=26.13%。按官方公式，mAP每增加1个百分点约增加1.9135 Δm，F1每增加1个百分点约增加0.5358 Δm；故同等百分点mAP权重约3.57倍。不能为了小F1收益牺牲较多mAP。

Phase1每天5次提交；Phase2全程2次。Phase1截止11月1日23:59 SGT，Phase2为11月2日00:01至11月8日23:59。选择公开posted成绩；没有posted时会取该阶段最后一次成功提交，并不自动选择所有私有历史最佳。小于约2 Δm的收益，规则要求跨seed均值及标准差；目前+1.29不能称稳定收益。以[当前官方页面](https://web-production-c241e.up.railway.app/docs)为准。

## 本地证据

完整数字及源文件SHA256在[data_audit.json](data_audit.json)。基线为`BP7_LR130_local_s0_20261029_v2/lr1x`，训练7512，验证940，seed0；下列类AP使用统一submission后处理。

| 类别 | AP | 训练框中位sqrt面积，512尺度 | 小于16像素比例 |
|---|---:|---:|---:|
| bridge |4.78%|9.91|82.8%|
| playground |22.94%|14.88|64.0%|
| ship |39.14%|55.71|0.5%|
| wind_turbine |72.98%|8.16|94.2%|

图像分类48个错误中24个是bridge↔playground。弱类并非简单样本少：bridge999张训练图、playground939张，car只有355张却AP62.75%。bridge长短边比>3仅3.3%，playground仅0.1%；扩展极端anchor长宽比不是它们的首选。wind_turbine更小却明显更好，尺寸也不是唯一原因。

### 新增CPU错误分解

使用[TIDE论文](https://arxiv.org/abs/2008.08115)及[官方实现](https://github.com/dbolya/tide)1.0.1，分析已有`predictions_val.json`。这是训练时原生后处理，**不是submission后处理**，没有额外前向或官方val标签。AP50=73.245955、AP75=56.319134，与原COCO评估一致到数值精度。

| 正匹配IoU | AP | 漏检Miss的oracle dAP | 定位Loc的oracle dAP | 背景Bkg的oracle dAP |
|---|---:|---:|---:|---:|
|0.50|73.246|14.736|3.461|1.341|
|0.75|56.319|9.323|20.437|0.741|

结果支持优先调查漏检和精细定位。oracle dAP表示假设理想修正某项错误的影响，不能相加，也不代表某个新算法可实际取得该增益。最终输出漏检也不等于RPN必定没有proposal，分数筛选/NMS等也可能参与。检测分类错误很小不表示图像分类没有问题，它们是不同任务。

详细结果：[tide_errors.json](tide_errors.json)。TIDE没有改入训练环境，依赖安装在`/tmp/cs701_autoresearch_tide`。

### 避免重复的旧实验

下表是旧r32/head1024的36轮筛选，不可与当前r80/head1536的最终AP直接相减。

| 方法 | val mAP | test mAP | 解释 |
|---|---:|---:|---|
|AA基线|38.429|40.009|旧筛选基线|
|context3|39.446|40.841|有单因素收益|
|仅分类context3|39.816|41.329|与当前BP7组合未验证|
|spatial64（BP7）|40.017|41.366|已采用|
|弱类oversample2|36.724|37.732|不支持重试同配置|
|partial negatives|38.524|40.122|接近持平|

来源：`BP_queue/followup_20261028/comparison.json`等记录。AA提高RPN到2000/3000候选仅有约0.01pp的val变化且test略降；SC1框投票和解除全图100框上限也无整体收益。EMA0.9998旧同run对照val mAP比last低0.221pp、test低0.024pp。IoU质量分支、D4/copy-paste、zoom已有探索，不应称全新方向。详见相邻实验README。

## 候选实验排序

### A：BP7 + 仅分类分支的上下文（第一训练候选）

保持原骨干/学习率/日程，只添加`--roi-context 3 --roi-context-mode classification`。分类使用较大区域，回归仍用局部区域。理由是旧BP8已有正面证据，而bridge/playground容易混淆、背景语义可能有用。与[TSD/Sibling Head论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Song_Revisiting_the_Sibling_Head_in_Object_Detector_CVPR_2020_paper.html)解耦分类和定位的动机相近，但本实现不是TSD，不能照搬其收益。

当前实现每阶段增加一个256→64的1×1卷积，共49,344参数，预计训练参数39,916,704、总参数125,126,304，均在预算内；真正启动前仍须实测统计。仅切换已有context的mode才是0新增参数。额外跑RoI head会增加计算，不能称零成本。先测是否过拟合背景及回归是否受益，收益不保证与spatial64相加。

### B：诊断驱动的NWD排序匹配（第二训练候选）

当前实现是`max(IoU,NWD)`加固定Matcher，尚非[NWD-RKA](https://arxiv.org/abs/2206.13996)的ranking assignment。先统计最新BP7中每GT的正anchor/正RoI数、排除注入GT后的proposal覆盖、各Cascade阶段目标流失，再决定只改RPN匹配还是Cascade阈值。

最小对照仅替换一个assigner，其他保持固定；参考NWD-RKA或[ATSS](https://arxiv.org/abs/1912.02424)的每GT自适应分配，不能同时改采样、损失和neck。训练时匹配不新增推理参数。风险是小目标密集时正样本冲突和背景误检。旧SC1的bridge/proposal覆盖低是方向证据，不是最新BP7的直接测量。

### C：Cascade的定位损失（根据TIDE结果值得研究）

固定assigner后，再比较现有encoded-box Smooth-L1与加入decoded-box GIoU的方案。优先检查AP75/AP90、每类定位误差和小框稳定性；不先同时加质量分支。参考[GIoU论文](https://arxiv.org/abs/1902.09630)，该损失无新增推理参数，但对极小框未必稳定。

实现注意：现有`--giou`在Cascade模式明确不支持，不能直接打开参数；需要给Cascade逐阶段实现并验证坐标、正样本、空样本和归一化。不能声称现成可跑。

### D：训练策略备选，保持峰值LR

1. **按更新步数对齐日程**。local130=61,100步；local160=75,200；full130=76,310；full160=93,920。本地160与全量130只差约1.5%，因此本地更长日程有益并不能直接证明全量也需160。下次比较固定step预算的余弦/低LR尾部，不盲目加到200轮。该发现不改变已向用户说明并启动的全量选型队列。
2. **归一化/偏置不做weight decay**。代码已有`--wd-hygiene`，当前默认仍对所有训练参数施加0.05衰减。可作为不改变架构与峰值LR的单因素对照，但目前没有本地正面结果支持，优先级低于A/B。不要把分类下降直接归因于weight decay。
3. **后处理小范围校准**。在最新local BP7的同一前向输出上固定少量分类融合λ/定位阶段设置，验证是否保住F1且不损mAP。已有简单分类概率乘检测分数、NMS/框投票弱收益，拒绝大网格官方试分。

通用教师蒸馏、跨层LoRA秩重分配是较远期方向，尚无本地证据；教师预算、参数并集和成本需另核查。EMA/SWA先确认规则，预测集成则不做。

## 执行顺序与停止条件

1. 当前160轮→已授权full train→submission流水线保持运行。
2. 本轮CPU研究已经完成，不占训练GPU，不上传。
3. 若全量结果仍值得继续投入，下一组优先A；只试一个变量，固定本地划分、seed及完整日程。不在full模型已见过的本地940/940上声称泛化提升。
4. 小收益需要配对seed复验。提交额度用于有本地证据的候选，Phase2只保留确定方案。不要把某个早期epoch的小优势当成最终收益。

复现CPU审计：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/tmp/cs701_autoresearch_tide MPLCONFIGDIR=/tmp/cs701_autoresearch_mpl \
TIDE_PATH=/tmp/cs701_autoresearch_tide_cache \
~/miniconda3/envs/cs701bench/bin/python tools/research_bp7_errors.py
```
