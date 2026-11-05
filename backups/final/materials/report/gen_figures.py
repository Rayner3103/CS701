#!/usr/bin/env python3
"""Charts for the report, one PNG per language. Numbers: results/*.json where they exist, otherwise ledger values cited in the report."""
import glob, json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
HERE = Path(__file__).resolve().parent; RES = HERE.parent / 'experiments/report_final_20261105/results'; OUT = HERE / 'figures'; OUT.mkdir(exist_ok=True)
fm.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
BLUE, ORANGE, INK, MUT, GRID, SURF, TINT, GRAY = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#ffffff', '#dbe8f8', '#ececea'
CLS_EN = {1:'aircraft',2:'airport',3:'bridge',4:'car',5:'harbor',6:'oil tank',7:'playground',8:'ship',9:'wind turbine'}
CLS_ZH = {1:'飞机',2:'机场',3:'桥梁',4:'汽车',5:'港口',6:'油罐',7:'操场',8:'船只',9:'风机'}
def setup(lang):
    plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'] if lang == 'zh' else ['DejaVu Sans'], 'font.size': 9, 'axes.edgecolor': MUT, 'axes.labelcolor': MUT, 'xtick.color': MUT, 'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6, 'savefig.dpi': 200, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.unicode_minus': False})
def save(fig, name, lang):
    fig.tight_layout(); fig.savefig(OUT / f"{name}{'.zh-CN' if lang == 'zh' else ''}.png", bbox_inches='tight', pad_inches=.06); plt.close(fig)
def T(lang, en, zh): return zh if lang == 'zh' else en
def grid(ax, axis='y'): ax.grid(axis=axis, color=GRID, lw=.6); ax.set_axisbelow(True); ax.tick_params(length=0)
def model(lang):
    t = lambda a, b: T(lang, a, b)
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.set_xlim(-1.5, 101.5); ax.set_ylim(1, 82); ax.axis('off')
    def box(x, y, w, h, s, c=TINT, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=1.0', fc=c, ec='#8a8984', lw=.7)); ax.text(x + w / 2, y + h / 2, s, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.25)
    def arr(a, b): ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', lw=.9, color=MUT, mutation_scale=9, shrinkA=0, shrinkB=0))
    box(0, 36, 14, 16, t('Gray image\n512 x 512', '灰度图\n512 x 512'), SURF)
    box(20, 46, 30, 34, '', GRAY); ax.text(35, 72, t('DINOv3 ViT-B/16\nfrozen, 85.1 M', 'DINOv3 ViT-B/16\n冻结，85.1 M'), ha='center', va='center', fontsize=8.5, color=INK, linespacing=1.25); box(23, 49, 24, 14, t('LoRA r80 + norms\n11.8 M', 'LoRA r80 + 归一化层\n11.8 M'))
    box(20, 12, 30, 16, t('Spatial prior CNN\n64 channels, 0.21 M', '空间先验 CNN\n64 通道，0.21 M'))
    box(56, 34, 17, 26, t('ViTDet pyramid\nstrides 4-64\n5.56 M', 'ViTDet 金字塔\n步长 4-64\n5.56 M'), fs=8)
    box(56, 66, 44, 12, t('Mean token: image class head, 0.01 M', '平均 token：图像分类头，0.01 M'))
    box(78, 34, 22, 26, t('RPN, 0.60 M\nanchors 6-181\nmax(IoU, NWD)', 'RPN，0.60 M\n锚框 6-181\nmax(IoU, NWD)'), fs=8)
    box(56, 4, 44, 18, t('3-stage Cascade, width 1536\nsize-aware IoU, 21.65 M', '三阶段 Cascade，宽度 1536\n尺寸感知 IoU，21.65 M'))
    arr((14, 48), (20, 58)); arr((14, 40), (20, 24)); arr((50, 72), (56, 72)); arr((50, 54), (56, 50)); arr((50, 22), (56, 38)); arr((73, 47), (78, 47)); arr((89, 34), (89, 22)); arr((64.5, 34), (64.5, 22))
    save(fig, 'model', lang)
def official(lang):
    pts = [('V2\n10-20', 38.23), ('F1\n10-21', 41.26), ('F2\n10-27', 44.84), ('SC2\n10-28', 51.86), ('BP7 cos160\n10-31', 52.78), ('cos200 e160\n11-01', 53.41), ('e180 + PP\n11-01', 54.83), (T(lang, 'avg x4', '平均 4 个') + '\n11-01', 55.41), (T(lang, 'avg x9\nhard', '平均 9 个\n硬融合') + '\n11-01', 56.41), (T(lang, 'avg x9\nclass-agn.', '平均 9 个\n类无关') + '\n11-01', 56.49)]
    fig, ax = plt.subplots(figsize=(7.2, 3.1)); x = range(len(pts)); y = [p[1] for p in pts]
    ax.plot(x, y, color=BLUE, lw=2, marker='o', ms=5, mec=SURF, mew=1); grid(ax); ax.set_xticks(list(x)); ax.set_xticklabels([p[0] for p in pts], fontsize=7); ax.set_ylabel(T(lang, 'Official val Δm', '官方验证集 Δm')); ax.set_ylim(35, 60)
    for i in (0, 2, 3, len(pts) - 1): ax.annotate(f'{y[i]:.2f}', (i, y[i]), textcoords='offset points', xytext=(-7, 8), ha='right' if i in (2, 3) else 'center', fontsize=8, color=INK)
    save(fig, 'official_progress', lang)
def ladder(lang):
    runs = [json.load(open(f)) for f in sorted(glob.glob(str(RES / 'pp_ablation_*.json')))]
    keys = ['+ sigma 0.5', '+ objectness power 0', '+ RPN 2000', '+ lambda 2', '+ average e100-180', '+ hard fusion', 'agn fusion instead', '+ min_keep 100', '+ lattice snap (FINAL)']
    lab = {'en': ['σ 0.75 → 0.5', 'objectness power 0', 'RPN top 2000', 'λ 2', 'average e100–180', 'hard fusion', 'class-agnostic vs hard', 'min_keep 100', 'lattice snap'], 'zh': ['σ 0.75 → 0.5', 'objectness 指数 0', 'RPN 候选 2000', 'λ 2', '平均 e100–180', '硬融合', '类无关（对比硬融合）', 'min_keep 100', '网格吸附']}[lang]
    v = [[r['ladder_steps'][k]['dm'] for r in runs] for k in keys]; m = [st.mean(a) for a in v]; e = [st.stdev(a) if len(a) > 1 else 0 for a in v]
    if len(runs) == 1: ci = [runs[0]['ladder_steps'][k]['dm_ci'] for k in keys]; err = [[m[i] - ci[i][0] for i in range(9)], [ci[i][1] - m[i] for i in range(9)]]
    else: err = e
    fig, ax = plt.subplots(figsize=(7.2, 3.2)); yy = list(range(len(keys)))[::-1]
    ax.barh(yy, m, height=.55, color=BLUE, xerr=err, error_kw=dict(ecolor=MUT, lw=.8, capsize=2)); ax.axvline(0, color=MUT, lw=.7); grid(ax, 'x'); ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel(T(lang, f'Δm gain of each cumulative step on vallike (n={len(runs)} seed; bars show ' + ('95% bootstrap CI)' if len(runs) == 1 else '±1 std)'), f'累积每一步的 Δm 增益（vallike，n={len(runs)}；误差线为 ' + ('95% bootstrap 区间）' if len(runs) == 1 else '±1 标准差）')), fontsize=8)
    save(fig, 'pp_ladder', lang)
def loo(lang):
    f = RES / 'loo.json'
    if not f.exists(): return
    V = json.load(open(f))['variants']; b = V.get('base')
    names = {'noSP': ('no spatial prior', '去掉空间先验'), 'noSizeIoU': ('no size-aware IoU', '去掉尺寸感知 IoU'), 'noFlipBias': ('no flip bias', '去掉翻转偏置修正'), 'noNWD': ('no NWD matching', '去掉 NWD 匹配'), 'noCascade': ('no Cascade', '去掉 Cascade'), 'noSmallAnchors': ('reference anchors', '改回参考锚框'), 'noTrainNorms': ('LayerNorms frozen', '冻结 LayerNorm'), 'smallCapacity': ('LoRA r32, head 1024', 'LoRA r32，头 1024')}
    rows = []
    for k, n in names.items():
        ds = [V[k]['dm'][V[k]['seeds'].index(s)] - b['dm'][b['seeds'].index(s)] for s in V[k]['seeds'] if s in b['seeds']] if k in V else []
        rows.append((n[1] if lang == 'zh' else n[0], st.mean(ds) if ds else None, st.stdev(ds) if len(ds) > 1 else 0, len(ds)))
    rows.sort(key=lambda r: (r[1] is not None, r[1] if r[1] is not None else 0), reverse=True); rows = rows[::-1] if False else rows
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); y = list(range(len(rows)))[::-1]; done = [v[1] for v in rows if v[1] is not None]
    lo = min(done + [-3]) - 1.1; hi = max(done + [1.5]) + .9
    ax.axvspan(-.70, .70, color=GRAY, lw=0, zorder=0, label=T(lang, 'seed noise of C36, ±0.70', 'C36 的 seed 噪声，±0.70'))
    for yy, r in zip(y, rows):
        if r[1] is None: ax.text(lo + .08, yy, T(lang, 'TBD', '待定'), va='center', fontsize=8, color=MUT)
        else:
            ax.barh(yy, r[1], height=.55, color=BLUE, xerr=r[2] or None, error_kw=dict(ecolor=MUT, lw=.8, capsize=2))
            ax.text(r[1] + (.08 if r[1] >= 0 else -.08) + (r[2] if r[1] >= 0 else -r[2]), yy, f"{r[1]:+.2f} (n={r[3]})", va='center', ha='left' if r[1] >= 0 else 'right', fontsize=7.5, color=INK)
    ax.axvline(0, color=MUT, lw=.7); grid(ax, 'x'); ax.set_xlim(lo, hi); ax.set_ylim(-.7, len(rows) - .4); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax.set_xlabel(T(lang, 'Δm of the variant minus the final recipe, 36-epoch runs on vallike (error bars ±1 std where n > 1)', '变体减最终配方的 Δm，36 轮，vallike（n > 1 时误差线为 ±1 标准差）'), fontsize=8)
    save(fig, 'loo', lang)
def iou(lang):
    thr = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]; val = [74.6, 73.7, 71.6, 68.8, 64.5, 58.2, 49.7, 37.4, 22.4, 8.1]; test = [75.0, 73.9, 72.3, 69.3, 64.9, 59.8, 50.9, 38.2, 23.3, 10.2]
    fig, ax = plt.subplots(figsize=(4.6, 2.9)); a = T(lang, 'local val', '本地验证集'); b = T(lang, 'local test', '本地测试集')
    ax.plot(thr, val, color=BLUE, lw=2, marker='o', ms=4, mec=SURF, label=a); ax.plot(thr, test, color=ORANGE, lw=2, marker='s', ms=4, mec=SURF, label=b); grid(ax); ax.legend(frameon=False, fontsize=8); ax.set_xlabel(T(lang, 'IoU threshold', 'IoU 阈值')); ax.set_ylabel('AP'); ax.set_ylim(0, 85)
    save(fig, 'ap_vs_iou', lang)
def perclass(lang):
    r = json.load(open(RES / 'pp_ablation_TWIN1_s0.json')); pc = next(x for x in r['results'] if x['name'].startswith('+ lattice'))['per_class']
    names = CLS_ZH if lang == 'zh' else CLS_EN; it = sorted(((names[int(k)], v) for k, v in pc.items()), key=lambda a: a[1])
    fig, ax = plt.subplots(figsize=(4.6, 2.9)); y = range(len(it)); ax.barh(list(y), [i[1] for i in it], height=.55, color=BLUE); grid(ax, 'x'); ax.set_yticks(list(y)); ax.set_yticklabels([i[0] for i in it], fontsize=8); ax.set_xlabel(T(lang, 'AP on vallike, FINAL, seed 0', 'vallike 上的 AP，FINAL，seed 0'), fontsize=8)
    for i, a in zip(y, it): ax.text(a[1] + 1, i, f'{a[1]:.0f}', va='center', fontsize=7.5, color=INK)
    ax.set_xlim(0, 98); save(fig, 'per_class_ap', lang)
def curve(lang):
    e = list(range(10, 200, 10)); m = [24.96, 31.65, 37.50, 41.07, 43.00, 44.63, 49.03, 50.44, 51.42, 53.01, 53.78, 54.54, 55.53, 55.75, 56.25, 56.13, 56.39, 56.28, 56.19]
    fig, ax = plt.subplots(figsize=(4.6, 2.9)); ax.plot(e, m, color=BLUE, lw=2); grid(ax); ax.axvspan(100, 180, color=TINT, zorder=0, lw=0); ax.text(140, 28, T(lang, 'averaging window\ne100–180', '平均窗口\ne100–180'), ha='center', fontsize=8, color=MUT); ax.set_xlabel(T(lang, 'Epoch (200-epoch cosine)', '轮数（200 轮余弦）')); ax.set_ylabel(T(lang, 'mAP on vallike', 'vallike mAP')); save(fig, 'training_curve', lang)
def sizes(lang):
    lab = ['<8', '8–16', '16–32', '32–96', '≥96']; v = [18.4, 32.1, 11.9, 30.8, 6.8]
    fig, ax = plt.subplots(figsize=(4.6, 2.9)); ax.bar(lab, v, width=.55, color=BLUE); grid(ax); ax.set_xlabel(T(lang, 'Box side at the 512 px input (px)', '512 输入下的框边长（像素）')); ax.set_ylabel(T(lang, 'Share of boxes (%)', '框占比（%）'))
    for i, a in enumerate(v): ax.text(i, a + .8, f'{a:.1f}', ha='center', fontsize=8, color=INK)
    save(fig, 'box_sizes', lang)
def neardup(lang):
    lab = ['≥ .99', '.97–.99', '.94–.97', '< .94']; a = [83.0, 62.0, 52.0, 40.6]; b = [83.7, 60.1, 55.8, 39.9]
    fig, ax = plt.subplots(figsize=(4.6, 2.9)); x = range(4); w = .34
    ax.bar([i - w / 2 - .01 for i in x], a, width=w, color=BLUE, label=T(lang, 'local val', '本地验证集')); ax.bar([i + w / 2 + .01 for i in x], b, width=w, color=ORANGE, label=T(lang, 'local test', '本地测试集')); grid(ax); ax.set_xticks(list(x)); ax.set_xticklabels(lab); ax.legend(frameon=False, fontsize=8); ax.set_xlabel(T(lang, 'Cosine similarity to nearest training image', '与最近训练图的余弦相似度')); ax.set_ylabel('mAP'); save(fig, 'near_duplicates', lang)
def snap(lang):
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7)); tt = [T(lang, 'Integer cells: corners rounded to the pixel grid', '整数格：角点取整到像素网格'), T(lang, 'Wind turbines: grid shifted by the image offset', '风机：网格按该图的偏移平移')]
    off = [(0, 0), (.4, .7)]
    for ax, title, (ox, oy) in zip(axs, tt, off):
        ax.set_xlim(-.3, 6.3); ax.set_ylim(-.3, 5.3); ax.set_xticks([i + ox for i in range(6 + (ox == 0))]); ax.set_yticks([i + oy for i in range(5 + (oy == 0))]); ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True); ax.tick_params(length=0, labelbottom=False, labelleft=False); ax.set_title(title, fontsize=8.5, color=INK); ax.set_aspect('equal')
        for sp in ax.spines.values(): sp.set_visible(False)
    axs[0].add_patch(Rectangle((1.3, 1.4), 3.3, 2.2, fill=False, ec=ORANGE, lw=1.6, ls='--', label=T(lang, 'prediction', '预测框'))); axs[0].add_patch(Rectangle((1, 1), 4, 3, fill=False, ec=BLUE, lw=2, label=T(lang, 'after snapping', '吸附后')))
    for (x, y, w, h) in [(.6, 1.65, 1.8, 1.1), (3.15, 2.8, 1.9, 1.25)]:
        axs[1].add_patch(Rectangle((x, y), w, h, fill=False, ec=ORANGE, lw=1.6, ls='--'))
        x1, y1, x2, y2 = round(x - .4) + .4, round(y - .7) + .7, round(x + w - .4) + .4, round(y + h - .7) + .7
        axs[1].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=BLUE, lw=2))
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=7.5, loc='lower center', ncol=2, bbox_to_anchor=(.5, -.05)); save(fig, 'lattice_snap', lang)
for lang in ('en', 'zh'):
    setup(lang)
    for f in (model, official, ladder, loo, iou, perclass, curve, sizes, neardup, snap): f(lang)
print(sorted(p.name for p in OUT.glob('*.png')))
