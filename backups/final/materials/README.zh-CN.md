# CS701 Team 7 最终材料

[English](README.md) · **中文**

报告、代码和实验记录的汇总，截至 2026 年 11 月 8 日。报告补跑的全部 30 个训练任务已于 03:24 SGT 成功结束，三次运行的结果已写入报告。仍标注“待定”的项目缺少可核实记录。文档都有两个版本：`X.md` 是英文，`X.zh-CN.md` 是中文。

**从这里开始：[report/REPORT.zh-CN.md](report/REPORT.zh-CN.md)**（图文报告和完整实验日志）。

## 目录

| 路径 | 内容 | 是否推送 |
|---|---|---|
| `report/` | `REPORT.md`、`REPORT.zh-CN.md`、`figures/` 里的图表和生成脚本 | 是 |
| `code/` | 最终模型和报告实验的代码快照 | 是 |
| `experiment_records/` | 各实验的说明、配置、脚本和结果摘要；`backup_56.49/` 是 56.49 版本的复现说明和运行记录 | 是 |
| `run_records/` | 每次训练的配置、指标和日志 | 是 |
| `phase2_records/` | Phase 2 的计划、排队和评估脚本、TWIN1 结论 | 是 |
| `manifest/` | 未推送大文件的清单和校验值、训练进度、Phase 2 榜单快照 | 是 |
| `local_only/` | 权重、原始输出、提交文件、会话记录、实验台账（只在服务器上） | 否 |

## Phase 2 成绩

Δm **61.25**，macro-F1 96.96，mAP 55.74，AP50 75.19，11 月 3 日 15:37 SGT 发布。11 月 8 日 12:21 SGT 的公开榜单上排 9 队第 3；当时尚未锁定，当天 23:59 SGT 截止，因此不是最终排名。快照见 `manifest/leaderboard_phase2_2026-11-08.json`；获取时间见 `manifest/leaderboard_snapshot_meta.json`。

## 完成情况与剩余项目

| 项目 | 状态 |
|---|---|
| Phase 2 实际提交的文件和次数 | 待定：回忆中最后一次是 `PHASE2_shot1_FINAL_avg100-180x9_agn_mk100_SNAP.zip`，榜单不显示文件名 |
| 其他作者 | 待定 |
| 完整配方 seed 1、2 | 已完成，各 180 轮；连同 seed 0，共三个全长本地运行；见 `manifest/training_status.txt` |
| 留一消融（8 个变体，3 个 seed） | 已完成：含三个配对基线，共 27 次运行，每次 36 轮；见表 2 |
| 本地参考模型 | 已完成：24 轮；默认评估下 F1 92.31、mAP 25.66、AP50 54.81 |
| 全项目 GPU 时总数 | 约 514 = 历史约 388 + 报告队列 126.31；队列时间包含预测和平均，不含队友的 RTX 3080 运行；见 `manifest/report_compute.json` |
| 报告格式和截止日期 | 待定：“CVPR 模板、不超过 6 页”只来自队内草稿，目前不受页数限制；LaTeX 版暂缓 |

## 没有推送的内容及原因

仓库是公开的，Terms 不允许向队外分享数据和预测，Phase 2 也还没有结束。

- **预测和数据：** 提交 zip、含逐图框的 JSON、带标签的划分文件。出于同样原因，报告的图表里没有 SAR 图像。
- **权重和原始输出：** 单个文件最大超过 600 MB，超出 GitHub 的限制。见 `manifest/large_files_not_pushed.tsv` 和 `manifest/*.sha256`。
- **会话记录和实验台账：** 含聊天原文和内部讨论。
- **56.49 备份的旧报告：** 部分内容已过时，以 `report/REPORT.zh-CN.md` 为准。

不翻译的内容：日志、数据文件，以及第三方的 `LICENSE`、`NOTICE`、`LEGAL`。

## 重建

在本材料目录内重新生成报告（使用报告的 Python 环境）：

```bash
python report/gen_figures.py
python report/gen_report.py
```

在保留实验源文件的原服务器上重建整个材料包：

```bash
report_final/build_materials.sh   # 在仓库根目录运行
```
