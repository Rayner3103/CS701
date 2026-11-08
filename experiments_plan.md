# 下一步实验清单 (CS701, 截至 2026-10-23)

现状盘点（见 USAGE.md / dino_yolo.md）:
- **主线 (Junkai)**: DINOv3-B/16 冻结 + LoRA r32 全线性 + norms → ViTDet FPN →
  NWD-RPN → Cascade R-CNN + 精调后处理. Codabench val **Δm 38.23** (V2_e100).
  `F1_full` = 同配方全量数据 100ep 已训完，Phase 2 候选 #1。
- **YOLO 线 (Rayner/你)**: DINOv3-S 融合 YOLO12s/26s，本地 test Δm ≤ +21.3;
  你的 YOLO11s Codabench val 20.53。定位：report 里的 efficiency 对比项。

## A. 服务器上跑（主线，Phase 2 前的最后窗口）

Phase 2 只有 **2 次提交**，11/2 拿到 test 图。目标：让两次提交覆盖
「最稳的一个 + 一个有上界的变体」。按期望收益排序：

1. **F1_full + EMA 变体** — train.py 有 `--ema 0.999`，几乎白送 ~+0.3-1 mAP。
   若 F1_full 当时没开 EMA，可用相同配置加 `--ema` 重训一版（3.5h）。
2. **第二 seed 的 F1_full** (`--seed 1`) — 对冲单次训练噪声 (±1 Δm)，
   两次提交 = seed0 + seed1，选 leaderboard 上表现好的那个。
3. **`--cascade-size-iou`** — 按 GT 尺寸分级 IoU 阈值，tiny 框在 stage 2/3
   才有正样本。已在 train.py 实现未用过，36ep screening 有正收益就上。
4. **`--iou-branch 1.0` + `quality_pow`** — IoU 质量分支重打分，免费 flag。
5. **`--unfreeze-last 1`** — 解冻最后一块 (7.1M)，23.8+7.1=30.9M 仍在预算内。
   full-FT 上限 +13.5 说明容量还有空间。
6. **`--flip-bias`** 训练时对齐标注偏移，替代后处理 −0.4px shift（更干净）。

screening 协议（照 USAGE.md §5）：train+local-val 训练 36ep → `pp_eval.py`
打分 → 与 C36 (val mAP 37.22 / F1 95.54) 比，>+1 才算真收益。

## B. 本机 RTX 5060 8GB 能跑的（已实测 2026-10-23）

**已验证**：完整配方（DINOv3-B + LoRA r32 全线性 + norms + Cascade + NWD +
小锚点）在本机跑通，参数数与文档一致 109,014,752 / 23,805,152。
batch 4 + `--grad-ckpt` 峰值仅 3.3 GB、~1.1 img/s；batch 2 是 2.9 GB。
→ 全量 100ep 训练不现实（~190h），但以下都可做：

1. **短 run / 调试**：24-36ep 的 screening 子集（`--limit`）验证 flag 改动。
2. **推理/评估**：无反向传播，batch 8+ 轻松；`predict.py` 生成提交 zip、
   `pp_eval.py`/`pp_study.py` 复调后处理全在本机可行（要拷 raw_*.pt）。
3. **YOLO 线补实验**（report 对比 + bonus「interesting extension」）:
   - `train_yolo.py --yolo yolo26s --epochs 100` 已在 3080 上 4h → 本机 ~6-8h，
     或 `--fraction` 先做短 run。
   - `--cls-head` 变体：F1 +3-4 / mAP −1.5 的 trade-off，可写进 report。
   - **frozen-DINO + YOLO26m**（dino_yolo.md「Not done」方向）：
     20M 可训练、更大检测主干，8GB 装得下。
4. **数据分析/图表**：box 尺寸分布 ✓(已出图)、per-class AP、训练曲线。
5. **提交文件体检**：`tools/check_submission.py` 本地校验 zip 格式。

环境备注：本机 Python 3.13 + torch 2.11.0+cu128 直接可用（非 repo 推荐的
3.12/2.9，但 smoke test 全部通过）；`sarbench/backbones.py` 已把 terratorch
import 改成 lazy（只 TerraMind backbone 需要），TF32 开关已对齐原行为。
DINOv3 权重下载走 hf-mirror（官方端点慢）：timm/vit_small_patch16_dinov3
和 vit_base_patch16_dinov3 已缓存（commit 3bf4720…/c6a5fb7…）。

## C. Phase 2 提交策略（11/2–11/8, 仅 2 次）

- 提交 1：`F1_full` (model.pt + FINAL 后处理) — 稳。
- 提交 2：A 清单里 screening 收益最大的变体；若都没有 ≥+1，则 F1_full 的
  不同 checkpoint（如 ckpt_e90）或第二 seed。
- 两次都用 `--raw-out` 存原始输出 → 若第 1 次分不理想，可换后处理重打包
  而不用重跑模型。

## D. 需要向组里拿的东西

- `runs/` 下关键 run 的 metrics.json + raw_*.pt（C36/C48/V2/F1_full）：
  做 pp_study 复调、画 report 图、算 per-class/size-AP。
- 你们 Codabench 已提交记录（哪几个 zip、各自分数）：report 的 val 表用。
- 你的 YOLO11s run 的权重/metrics（如要复现或写进对比表）。
- F1_full 是否用了 EMA / flip-bias / 其他新 flag（USAGE.md 未提）。
