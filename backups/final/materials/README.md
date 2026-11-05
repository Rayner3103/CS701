# CS701 Team 7 最终材料

报告、代码和实验记录的汇总，截至 2026-11-05。标注“待定”的内容还没有结果，补训练完成后更新。

## 目录

| 路径 | 内容 | 是否推送 |
|---|---|---|
| `report/` | 报告 `main.pdf` 和 LaTeX 源文件（正文 + 完整实验日志附录），`gen_report.py` 生成表格和图，`build_materials.sh` 重建本目录 | 是 |
| `code/` | 当前工作区的代码快照，含未提交的修改：`train.py`、`predict.py`、`sarbench/`、`tools/`、`tests/` | 是 |
| `experiment_records/` | `experiments/` 下各实验的说明、配置、脚本和结果摘要；`backup_56.49/` 是 56.49 备份的复现说明和运行记录 | 是 |
| `run_records/` | `cs701_runs/` 下每次训练的配置、指标和日志 | 是 |
| `phase2_records/` | Phase 2 的计划、排队脚本、评估脚本和 TWIN1 结论 | 是 |
| `manifest/` | 未推送大文件的清单和校验值，以及训练进度 `training_status.txt` | 是 |
| `local_only/` | 权重、原始输出、提交文件、会话记录、实验台账（硬链接，只在服务器上） | 否 |

## 待定

| 项目 | 状态 |
|---|---|
| Phase 2 test 成绩 | 待定：榜单目前连不上 |
| 实际提交的文件和次数 | 待定：目前依据回忆，最后一次应为 `PHASE2_shot1_FINAL_avg100-180x9_agn_mk100_SNAP.zip` |
| 其他作者姓名 | 待定 |
| 完整配方 seed 1、2 | 待定：训练中，见 `manifest/training_status.txt` |
| 留一消融（8 个变体，3 个 seed） | 待定：训练中 |
| 参考模型本地基准 | 待定：排队中 |
| 全项目 GPU 时总数 | 待定：截至 11-02 约 388，不含队友和补跑 |
| 报告格式要求和截止日期 | 待定：“CVPR 模板、不超过 6 页”只来自队内草稿；目前不受页数限制 |

## 没有推送的内容及原因

仓库是公开的，而 Terms 不允许向队外分享数据和预测；Phase 2 也还没有结束。

- **预测和数据：** 提交 zip、含逐图框的 JSON、带标签的划分文件。
- **权重和原始输出：** 单个文件最大超过 600 MB，超出 GitHub 的限制。清单见 `manifest/large_files_not_pushed.tsv`，提交文件和平均权重的校验值见 `manifest/*.sha256`。
- **会话记录和实验台账：** 含聊天原文和内部讨论。
- **56.49 备份的旧报告：** 其中的排名等内容已过时，以 `report/main.pdf` 为准。

## 重建

```bash
report_final/build_materials.sh   # 在仓库根目录运行，不会删除本 README
```
