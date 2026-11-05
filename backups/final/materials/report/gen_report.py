#!/usr/bin/env python3
"""Writes tables/*.tex and figures/*.pdf for main.tex from the experiment results. Numbers without a results file are
hand-checked against the ledgers in experiments/report_final_20261105/ledgers (sources in the supplement)."""
import glob, json, os, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = Path(__file__).resolve().parent; RES = HERE.parent / 'experiments/report_final_20261105/results'
(HERE / 'tables').mkdir(exist_ok=True); (HERE / 'figures').mkdir(exist_ok=True)
PEND = r'\textcolor{red}{TBD}'
cfg = json.loads((HERE / 'report_inputs.json').read_text())
def w(name, s): (HERE / 'tables' / name).write_text(s)
def pm(xs, d=2, sign=False):
    f = f'{{:+.{d}f}}' if sign else f'{{:.{d}f}}'
    if len(xs) == 1: return f.format(xs[0])
    return f.format(st.mean(xs)) + r'$\pm$' + f'{st.stdev(xs):.{d}f}'
def table(label, cap, cols, head, rows, star=False):
    env = 'table*' if star else 'table'
    return '\\begin{%s}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{@{}%s@{}}\n\\toprule\n%s\\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n\\caption{%s}\n\\label{%s}\n\\end{%s}\n' % (env, cols, head, '\n'.join(r + r'\\' for r in rows), cap, label, env)
w('macros.tex', '\\newcommand{\\TEAMMATES}{%s}\n\\newcommand{\\PTWO}{%s}\n\\newcommand{\\GPUH}{%s}\n' % (cfg['teammates'], cfg['phase2_score'], cfg['gpu_hours']))
# ---------------- official progression
rows = [r'10-20 & V2, 120-ep cosine, e100 & 38.23 & 95.06 & 45.62 & first long run, 90\,\% data',
 r'10-21 & F1, all data, 100 ep & 41.26 & 95.90 & 46.97 & all data, better post-proc.',
 r'10-27 & F2 & 44.84 & 95.46 & 48.97 & size-aware IoU, flip bias, shift',
 r'10-28 & SC2, e130 of 150 & 51.86 & 95.35 & 52.66 & LoRA r80, head 1536, longer',
 r'10-31 & BP7, e130 of 160 & 52.78 & 95.35 & 53.15 & spatial prior, slower cosine',
 r'11-01 & cos200, e160 & 53.41 & 95.02 & 53.56 & slower cosine again',
 r'11-01 & e180, $\sigma$.5, obj$^0$, RPN 2k, $\lambda$2 & 54.83 & 95.09 & 54.29 & post-processing only',
 r'11-01 & avg e100/120/140/160 & 55.41 & 95.20 & 54.56 & checkpoint averaging',
 r'11-01 & avg e100--180 (9), hard & 56.41 & 95.27 & 55.06 & wider window + hard fusion',
 r'\textbf{11-01} & \textbf{same, class-agnostic} & \textbf{56.49} & \textbf{95.27} & \textbf{55.10} & \textbf{posted, Phase 1}',
 r'\textbf{11-03} & \textbf{test: + min\_keep 100, snap} & \textbf{\PTWO} & \textbf{96.96} & \textbf{55.74} & \textbf{posted, Phase 2} (test reference)']
w('official.tex', table('tab:official', r'Official scores (Phase 1 val, reference 93.31 / 26.13; the last row is test, reference 93.99 / 25.41; seed 0, 1 run each). V2 and F1 were scored on Codabench, the rest on the course leaderboard.', 'lp{2.35cm}rrrp{2.0cm}', r'Date & Entry & \dm & F1 & mAP & Change', rows))
# ---------------- post-processing ablation over seeds
files = sorted(glob.glob(str(RES / 'pp_ablation_*.json'))); runs = [json.load(open(f)) for f in files]
if runs:
    names = ['old PP, e180', '+ sigma 0.5', '+ objectness power 0', '+ RPN 2000', '+ lambda 2', '+ average e100-180', '+ hard fusion', 'agn fusion instead', '+ min_keep 100', '+ lattice snap (FINAL)']
    label = {'old PP, e180': r'e180, $\sigma$.75, obj$^{.25}$, RPN 1k, $\lambda$1', '+ sigma 0.5': r'$\sigma$ 0.5', '+ objectness power 0': r'objectness power 0', '+ RPN 2000': r'RPN top 2000', '+ lambda 2': r'$\lambda$ 2', '+ average e100-180': r'\textbf{average e100--180}', '+ hard fusion': r'hard fusion', 'agn fusion instead': r'class-agnostic (vs hard)', '+ min_keep 100': r'min\_keep 100 (vs class-agn.)', '+ lattice snap (FINAL)': r'\textbf{snap = FINAL}'}
    rows = []
    for n in names:
        rr = [next(x for x in r['results'] if x['name'] == n) for r in runs]
        if n in runs[0]['ladder_steps']:
            sd = [r['ladder_steps'][n]['dm'] for r in runs]; sm = [r['ladder_steps'][n]['mAP'] for r in runs]; gain, gm = pm(sd, 2, True), pm(sm, 2, True)
        else: gain, gm = '---', '---'
        rows.append(f"{label[n]} & {pm([x['mAP'] for x in rr])} & {pm([x['F1'] for x in rr])} & {gm} & {gain}")
    loo = {'FINAL single e180': 'FINAL with e180 alone', 'FINAL without snap': 'without snap', 'FINAL no fusion': 'no fusion', 'FINAL objectness 0.25': r'objectness$^{.25}$', 'FINAL sigma 0.75': r'$\sigma$ 0.75', 'FINAL last-stage box': 'last-stage box', 'FINAL shift 0': 'shift 0'}
    rows2 = [f"{v} & {pm([-r['final_minus_variant'][k]['mAP'] for r in runs], 2, True)} & {pm([-r['final_minus_variant'][k]['dm'] for r in runs], 2, True)}" for k, v in loo.items()]
    cap = f"Cumulative post-processing and averaging on full-length runs (n={len(runs)} seed{'s' if len(runs) > 1 else ''}; \\texttt{{trainvl}} $\\to$ \\texttt{{vallike}}). Step gains are paired on one seed's weights; \\dm{{}} uses the val reference, on a local split, so its absolute level is not comparable to the board."
    s = table('tab:pp', cap, 'lrrrr', r'Step (cumulative) & mAP & F1 & $\Delta$mAP & $\Delta$\dm', rows)
    w('pp.tex', s)
else:
    w('pp.tex', '\\begin{table}[t]\\caption{%s}\\label{tab:pp}\\end{table}\n' % PEND)
# ---------------- leave-one-out
lf = RES / 'loo.json'
order = [('base', 'Final recipe'), ('noSP', 'no spatial prior'), ('noSizeIoU', 'no size-aware IoU'), ('noFlipBias', 'no flip bias (shift $-0.4$)'), ('noNWD', 'no NWD matching'), ('noCascade', 'no Cascade (Faster R-CNN)'), ('noSmallAnchors', 'reference anchors'), ('noTrainNorms', 'LayerNorms frozen'), ('smallCapacity', 'LoRA r32, head 1024')]
train = {'base': '39.87', 'noSP': '39.66', 'noSizeIoU': '39.87', 'noFlipBias': '39.87', 'noNWD': '39.87', 'noCascade': '32.16', 'noSmallAnchors': '39.86', 'noTrainNorms': '39.83', 'smallCapacity': '24.02'}
if lf.exists():
    v = json.load(open(lf))['variants']; rows = []
    b = v.get('base')
    for k, n in order:
        if k not in v: rows.append(f"{n} & {train[k]} & {PEND} & & & "); continue
        o = v[k]
        if k == 'base': g = '---'
        else:
            # paired by seed where both exist; otherwise difference of means
            common = [s for s in o['seeds'] if s in b['seeds']]
            ds = [o['dm'][o['seeds'].index(s)] - b['dm'][b['seeds'].index(s)] for s in common]
            g = pm(ds, 2, True) if ds else '---'
        rows.append(f"{n} & {train[k]} & {o['n']} & {pm(o['mAP'])} & {pm(o['F1'])} & {g}")
    w('loo.tex', table('tab:loo', r'Leave-one-out on \texttt{vallike} (36-epoch cosine, FINAL post-processing, last checkpoint). $\Delta$\dm{} is variant minus final recipe, paired by seed; mean$\pm$std where $n>1$.', 'lrrrrr', r'Variant & Train.\,M & $n$ & mAP & F1 & $\Delta$\dm', rows))
else:
    w('loo.tex', '\\begin{table}[t]\\caption{%s}\\label{tab:loo}\\end{table}\n' % PEND)
# ---------------- negative results
rows = [r'E09b wide LoRA + regularisers (24 ep) & raw local \dm{} 6.18 vs 18.08 & causes never separated',
 r'ConvNeXt-T full fine-tune (24 ep) & 6.78; AP$_{<8}$ 1.9 & below ViT + LoRA',
 r'Prior + multi-level neck (24 ep) & 5.14 & prior alone later won',
 r'D4 + copy-paste; long D4, zoom & 7.13; behind plain at all epochs & augmentation hurt',
 r'Overlapping patches, stride 8 & $-0.99$, 2.6$\times$ compute & oil tank $-6.8$, harbor $-4.6$',
 r'EMA 0.9998; IoU branch (36 ep) & $-0.13$; $-0.48$ & lead vanished as LR$\to$0',
 r'RoI batch 1024 & $-4.02$ & cause not studied',
 r'Bridge/playground: class weights, oversampling, partial-neg & bridge $\le+0.9$ AP; oversampling $-4.15$ & wind $-10$ AP',
 r'Classification-only RoI context (160 ep) & $+2.26$, $-0.12$, board $-0.03$ & did not transfer',
 r'LR $\times0.5$; head/backbone hybrid & $-7.21$; $-2.77$ & early lead lost by e100',
 r'LingBot-B backbone & board 39.23 at e60 (BP7 e80: 48.40) & paused, no fair pair',
 r'Box voting; no 100-box cap & val mAP $-0.03$ to $0$ & no effect',
 r'Rank, size priors; per-class bias tables; refinement & $-3.1$ to $+0.06$ mAP & noise or harm',
 r'kNN label transfer (local only) & $+0.52/+0.29$ mAP & test-time data, rules; not used']
w('negative.tex', table('tab:neg', r'Tried and dropped. Numbers are local \dm{} unless marked; screens are single-seed, so read small values with the $\pm0.70$ noise of \S\ref{sec:val} in mind.', 'p{7.2cm}p{5.4cm}p{4.2cm}', r'Idea & Result & Note', rows, star=True))
# ---------------- Terms
rows = [r'Parameters & \textbf{125,076,960 total, 39,867,360 trainable}: frozen ViT 85.14\,M; LoRA 11.80\,M; norms 0.04\,M; neck 5.56\,M + prior 0.21\,M; RPN 0.60\,M; Cascade 21.65\,M; class head 0.01\,M',
 r'GPU & 2 $\times$ NVIDIA RTX 6000 Ada (46,068\,MiB), shared with other users',
 r'Final training & 12.97\,h to e180 on 2 GPUs (25.9 GPU-h); the run continued to e200 at 14.14\,h. Averaging: 30\,s CPU',
 r'Project GPU-h & About \GPUH{} on this server, tuning included; RTX 3080 runs of teammates not counted',
 r'Peak GPU memory & Train 12.07\,GiB per GPU (gradient checkpointing, 8 per GPU); inference 1.15\,GiB allocated',
 r'Inference & 21--25\,ms per image at batch 16 (bf16, including post-processing, GPU shared); 60\,ms at batch 2 under load on the test run',
 r'Input & One gray image, stretched to $512{\times}512$; train mean/std 0.2526/0.2133; one pass, no tiling, no TTA',
 r'Pretrained & \textbf{Final:} DINOv3 ViT-B/16 LVD-1689M (timm). \textbf{Exploration only:} ImageNet-21k ViT-B/16; DINOv3 ConvNeXt-T; DINOv3 ViT-S/16 with YOLO; TerraMind; LingBot-Vision-B/16. None trained on SAR',
 r'Code & Base: TA reference repo; torchvision, timm, pycocotools. Ours: LoRA extensions, Cascade with size-aware IoU, NWD matching, spatial prior, post-processing, averaging, snapping, evaluator',
 r'Validation & \S\ref{sec:val}. Official val and test only predicted',
 r'Seeds & Submitted model: 1 run. Final recipe, full length: %s runs. Ablations: Table~\ref{tab:loo}. C36: 37 runs, $\pm0.70$' % cfg['twin_seeds']]
w('terms.tex', table('tab:terms', r'Items required by the Terms.', 'p{1.65cm}p{6.1cm}', 'Item & Value', rows))
# ---------------- model diagram (a diagram, not a chart: neutrals for frozen, one blue for trained)
BLUE, GRAY, INK, MUT = '#dbe8f8', '#ececea', '#0b0b0b', '#52514e'
fig, ax = plt.subplots(figsize=(3.4, 2.75)); ax.set_xlim(0, 100); ax.set_ylim(0, 82); ax.axis('off')
def box(x, y, w_, h, t, c=BLUE, fs=5.4):
    ax.add_patch(FancyBboxPatch((x, y), w_, h, boxstyle='round,pad=0.2,rounding_size=1.0', fc=c, ec='#8a8984', lw=0.5)); ax.text(x + w_ / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK, linespacing=1.1)
def arr(a, b): ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', lw=0.6, color=MUT, mutation_scale=5, shrinkA=0, shrinkB=0))
box(0, 36, 15, 16, 'Gray image\n512 x 512', '#ffffff')
box(21, 46, 30, 34, '', GRAY); ax.text(36, 72, 'DINOv3 ViT-B/16\nfrozen, 85.1 M', ha='center', va='center', fontsize=5.4, color=INK, linespacing=1.1); box(24, 49, 24, 14, 'LoRA r80 + norms\n11.8 M')
box(21, 12, 30, 16, 'Spatial prior CNN\n64 ch, 0.21 M')
box(57, 34, 16, 26, 'ViTDet\npyramid\nstrides 4-64\n5.56 M', fs=5.0)
box(57, 66, 43, 12, 'Mean token: class head')
box(78, 34, 22, 26, 'RPN 0.60 M\nanchors 6-181\nmax(IoU, NWD)', fs=5.0)
box(57, 4, 43, 18, 'Three-stage Cascade, width 1536\nsize-aware IoU, 21.65 M')
arr((15, 48), (21, 58)); arr((15, 40), (21, 24)); arr((51, 72), (57, 72)); arr((51, 54), (57, 50)); arr((51, 22), (57, 38)); arr((73, 47), (78, 47)); arr((89, 34), (89, 22))
fig.savefig(HERE / 'figures/model.pdf', bbox_inches='tight', pad_inches=0.01); plt.close(fig)
print('tables and figures written;', len(runs), 'twin seed file(s), loo', lf.exists())
