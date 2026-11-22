# Phase 2 夜间计划（由 Claude 写于 2026-11-01 21:15 SGT；用户已入睡）

[English](PLAN.md) · **中文**

## 状态
- Phase 1 最终结果：Δm 56.49 = avg(e100..e180 step 10，BP7 cos200 全数据运行的 9 个 ckpt) + rpn2000，obj_pow 0，soft-NMS σ0.5，stage mean，λ=2，shift −0.2，fusion agn γ=1 topk 9。权重：CS701/.runtime/avg/avg_e100-180.pt。同样配置但 fusion hard = 56.41。之前的最佳 55.41。
- Phase 2：test 于 11-02 00:01 发布，2,065 张图像，总共 2 次提交，截止 11-08 23:59 SGT。由用户手动提交。

## 正在运行的任务（全部用 setsid 脱离；会话重启后仍继续）
1. cs701_runs/AUG1_speckle_gamma_local7512_cos160_s0_20261101 — 光度增强的本地筛选（speckle max std 0.2 + gamma ±0.29，各自 p=0.5），使用 BP7_fixedLR160_local_s0_20261030_direct 的完全相同配方（7,512 张训练，cos160，stop e150，每 10 轮存 ckpt，每 10 轮评估 val）。日志：launch.log、smoke.log、schedule160.stdout.log。对照基线：BP7_fixedLR160_local_s0_20261030_direct/schedule160（ckpt e10..e150，log.txt 中每 10 轮有 val）。决策规则：对每个运行的 ckpt e80..e150 取平均，用 tools/boot_eval.py（配对 bootstrap）以最终后处理在两个本地留出集上打分。胜出 = 两个留出集上都 ≥ +0.3 mAP → 带增强的全数据 cos200 运行。
2. CS701/.runtime/phase2/watch_test_release.sh — 轮询 HF 上的 test*，下载并解压到 cs701_data/cs701-sar-course-data/test，构建 submissions/PHASE2_shot1_avg100-180x9_rpn2000_obj0_soft05_cls2_agn_g1_k9.zip（上述配方）和 PHASE2_backup_..._hardfuse.zip，校验格式。日志：.runtime/phase2/watch.log。完成时输出 "PHASE2 CANDIDATES READY"。
3. scratchpad boot/ 文件夹中的离线后处理研究（CPU）：boot_eval2（γ、σ、min_keep、thr、stage last）、lambda 扫描、stage 权重变体。

## 提交计划
- Shot 1（11-02 上午）：PHASE2_shot1 zip（安全，配方已在 Phase 1 验证）。发布它。
- Shot 2（11-07 之前）：本地验证最好的挑战者（用增强训练的全数据模型，平均方式和后处理相同），前提是它在两个本地留出集上都胜过基线。否则用同一权重的改进后处理，或者不提交。

## 遵守的规则
一个模型，每张图像在 512 下前向一次，无集成/TTA/切片，val/test 只做预测（从未用于训练/标注/在没有排行榜反馈时调参），对一条轨迹做参数平均（助教允许），总计 125.08M / 可训练 39.87M。
# 2026-11-01 21:36：GPU peer-to-peer（NCCL P2P/CUMEM）在这台主机上会挂起；所有 DDP 启动现在都设置 NCCL_P2P_DISABLE=1（已用 2-GPU all_reduce 测试验证）。

## 队列（截至 21:45 SGT）
AUG1（运行中，eager，NCCL_P2P_DISABLE=1，ETA ~04:30）-> [queue_after_aug1] DP1 drop-path 0.1 + AUG1 结论
-> [queue_after_dp1] IOU1（--iou-branch 1.0，先做烟测）+ DP1 结论 -> SA1（P2 anchors 5,7,10）+ IOU1 结论 -> SA1 结论。
结论文件：.runtime/phase2/eval_<tag>/verdict.txt（合并的 1,880 张图像代理集，uniform + val-like 权重，配对 bootstrap）。
代理工具：scratchpad proxy_eval.py；历史一致性检查：scratchpad/proxy/hist_eval.log。

## 代理集上的后处理发现（W3 本地权重，配对 bootstrap，2026-11-01 晚上）
- agn（soft fusion，gamma 1，topk 9）> hard > none：比 hard 高 +0.09/+0.15 mAP（val/test），官方 val 确认 +0.04（56.41 -> 56.49）。
- agn 下 sigma 0.4 对比 0.5：+0.01/+0.03（噪声水平，符号一致）；min_keep 100：+0.011/+0.015（P=1.0，单调）；sigma 0.4 + min_keep 100 同时使用：+0.022/+0.052（P .85/.98）。这是只用于 shot 2 的候选微调（shot 1 保持 56.49 的原配方）。
- 无增益：按类 sigma、stage 框权重、gamma 1.5/2、lambda != 2、score_thresh 1e-3、shift != -0.2、最后一个 stage 的框、mult fusion。
- 代理一致性：本地差值能跟踪官方差值（val 划分 corr 0.98，test 划分 corr 0.85）；类似 val 的类别/尺寸重加权并不能改善这一点。
- AUG1 之后排队的想法：'soup' = 基线运行和 AUG1 运行的平均权重再取平均（相同 seed/数据顺序）-> scratchpad/average_files.py。

## 协议变更 2026-11-01 21:50（用户：不要依赖旧的本地留出集；Terms 禁止外部 SAR 数据）
- 新留出集 `vallike`（992 张图像）= 旧本地 val+test 的子集，匹配官方 val 的（预测类别，尺寸）分布（由 scratchpad 代码构建；文件：dataset/SARFact-Course-20K/{classification/vallike.csv, detection/instances_vallike.json, vallike_split.json}）。`trainvl` = 其余 8,400 张官方训练图像。
- Twins = 最终配方（cos200，stop e190，每 5 轮存 ckpt）在 trainvl 上训练，每 10 轮在 vallike 上评估：TWIN1_base_trainvl8400_cos200_s0_20261101（自 21:49 起运行），然后是 TWIN2_aug_...（queue_twins.sh）。
- eval_twin.sh <run> <tag>：在 vallike 上用 proxy_eval.py --splits vallike 评估窗口 + 单点。
- vallike 与官方分数的一致性检查：6 个历史 7,512 模型（从未见过 vallike）-> scratchpad/proxy/hist_eval.log。
- 旧的 7,512 筛选（AUG1/DP1/IOU1/SA1）及其队列已停止。

## 2026-11-02 00:50 更新
- Phase 2 shot-1 候选：submissions/PHASE2_shot1_FINAL_avg100-180x9_agn_mk100_SNAP.zip（56.49 配方 + min_keep 100 + lattice snap）。本地类 test 估计的 snap+mk100 增益：+0.30 mAP = +0.59 Δm_test [+0.39,+0.80]。
- 今晚否决：stage 概率权重、迭代细化、w/h 取整、按单元格的偏差表、kNN 标签迁移（规则；用户拒绝）。
- Twin 链：TWIN1 base（ETA 07:30 11-02）-> TWIN2 giou3+iou1 -> TWIN3 aug -> [queue_twins2] TWIN4 P2 anchors 5,7,10 -> TWIN5 drop-path 0.1。eval_twin.sh 现在用完整的 FINAL 流程（--snap，min_keep 100）打分，存在 IoU 分支时再加 quality_pow 0.5/1/2。
- 时间预算：每次运行 ~10.5 h -> 11-08 之前大约能跑 13 次；计划：11-03/04 前选出胜者，对胜者做全数据运行，11-06 前提交 shot 2。
