# CS701 Team 7 最终材料

[English](README.md) · **中文**

报告、代码和实验记录的汇总，截至 2026 年 11 月 5 日。标注“待定”的内容还没有结果，训练结束后更新。文档都有两个版本：`X.md` 是英文，`X.zh-CN.md` 是中文。

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

Δm **61.25**，macro-F1 96.96，mAP 55.74，AP50 75.19，11 月 3 日 15:37 SGT 发布。11 月 5 日 18:59 的公开榜单上排 8 队第 3，榜单 11 月 8 日锁定。快照见 `manifest/leaderboard_phase2_2026-11-05.json`。

## 待定

| 项目 | 状态 |
|---|---|
| Phase 2 实际提交的文件和次数 | 待定：回忆中最后一次是 `PHASE2_shot1_FINAL_avg100-180x9_agn_mk100_SNAP.zip`，榜单不显示文件名 |
| 其他作者 | 待定 |
| 完整配方 seed 1、2 | 待定：训练中，见 `manifest/training_status.txt` |
| 留一消融（8 个变体，3 个 seed） | 待定：训练中 |
| 本地参考模型 | 待定：排队中 |
| 全项目 GPU 时总数 | 待定：截至 11 月 2 日约 388，不含队友和报告实验 |
| 报告格式和截止日期 | 待定：“CVPR 模板、不超过 6 页”只来自队内草稿，目前不受页数限制；LaTeX 版暂缓 |

## 没有推送的内容及原因

仓库是公开的，Terms 不允许向队外分享数据和预测，Phase 2 也还没有结束。

- **预测和数据：** 提交 zip、含逐图框的 JSON、带标签的划分文件。出于同样原因，报告的图表里没有 SAR 图像。
- **权重和原始输出：** 单个文件最大超过 600 MB，超出 GitHub 的限制。见 `manifest/large_files_not_pushed.tsv` 和 `manifest/*.sha256`。
- **会话记录和实验台账：** 含聊天原文和内部讨论。
- **56.49 备份的旧报告：** 部分内容已过时，以 `report/REPORT.zh-CN.md` 为准。

不翻译的内容：日志、数据文件，以及第三方的 `LICENSE`、`NOTICE`、`LEGAL`。

## 重建

```bash
report_final/build_materials.sh   # 在仓库根目录运行
```
