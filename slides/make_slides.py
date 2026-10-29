"""Build the CS701 talk draft (10 min + Q&A). One slide per section of outline.md.
Run: python make_slides.py  ->  CS701_talk_draft.pptx"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'report' / 'figures'

DARK = RGBColor(0x1B, 0x1B, 0x1B)
GREY = RGBColor(0x59, 0x59, 0x59)
ACCENT = RGBColor(0x0B, 0x5C, 0xAB)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1B, 0x7A, 0x3D)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def slide():
    return prs.slides.add_slide(BLANK)


def box(s, l, t, w, h):
    b = s.shapes.add_textbox(l, t, w, h)
    b.text_frame.word_wrap = True
    return b


def para(tf, text, size=18, bold=False, color=DARK, font='Calibri', space=4, first=False, align=None):
    p = tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()
    p.space_after = Pt(space)
    if align is not None:
        p.alignment = align
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = font
    return p


def title_bar(s, text, kicker=None):
    b = box(s, Inches(0.55), Inches(0.28), SW - Inches(1.1), Inches(0.9))
    if kicker:
        para(b.text_frame, kicker, 12, True, ACCENT, first=True)
    para(b.text_frame, text, 28, True, DARK, first=kicker is None)
    line = s.shapes.add_shape(1, Inches(0.55), Inches(1.14), SW - Inches(1.1), Emu(12700))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()
    return b


def bullets(s, items, l=0.75, t=1.35, w=None, size=17, gap=6):
    w = w or (13.333 - 2 * l)
    b = box(s, Inches(l), Inches(t), Inches(w), SH - Inches(t) - Inches(0.4))
    first = True
    for it in items:
        if isinstance(it, tuple):
            txt, lvl, bold = it
        else:
            txt, lvl, bold = it, 0, False
        p = para(b.text_frame, ('    ' * lvl) + txt, size - lvl, bold, DARK, space=gap, first=first)
        p.level = lvl
        first = False
    return b


def table(s, rows, l, t, w, h, col_w=None, size=14, header=True):
    nr, nc = len(rows), len(rows[0])
    g = s.shapes.add_table(nr, nc, Inches(l), Inches(t), Inches(w), Inches(h)).table
    if col_w:
        for i, cw in enumerate(col_w):
            g.columns[i].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = g.cell(i, j)
            c.text = str(val)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if j or (header and i == 0) else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.name = 'Calibri'
                    r.font.bold = header and i == 0
                    r.font.color.rgb = WHITE if header and i == 0 else DARK
            if header and i == 0:
                c.fill.solid()
                c.fill.fore_color.rgb = ACCENT
            elif i % 2 == 0:
                c.fill.solid()
                c.fill.fore_color.rgb = LIGHT
    return g


def note(s, text):
    s.notes_slide.notes_text_frame.text = text


def mono(s, text, l, t, w, h, size=13):
    b = box(s, Inches(l), Inches(t), Inches(w), Inches(h))
    b.text_frame.word_wrap = False
    for i, line in enumerate(text.split('\n')):
        para(b.text_frame, line, size, False, DARK, font='Consolas', space=1, first=i == 0)
    return b


def chip(s, text, l, t, w=2.9, color=ACCENT):
    sh = s.shapes.add_shape(5, Inches(l), Inches(t), Inches(w), Inches(0.55))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    tf = sh.text_frame
    tf.word_wrap = True
    para(tf, text, 14, True, WHITE, first=True, align=PP_ALIGN.CENTER)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


# ---------------- Slide 1: title ----------------
s = slide()
bg = s.shapes.add_shape(1, 0, 0, SW, SH)
bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(0x11, 0x1A, 0x26); bg.line.fill.background()
b = box(s, Inches(0.9), Inches(1.5), SW - Inches(1.8), Inches(4))
para(b.text_frame, 'Multi-Task Learning on SAR Images', 44, True, WHITE, first=True)
para(b.text_frame, 'A frozen DINOv3 backbone + LoRA, with a detector rebuilt for sub-patch objects', 22, False, RGBColor(0x9F, 0xC1, 0xE8))
para(b.text_frame, '', 12)
para(b.text_frame, 'Validation Δm +38.23   ·   macro-F1 95.06   ·   mAP 45.62', 20, True, RGBColor(0x8B, 0xE3, 0xA8))
b = box(s, Inches(0.9), Inches(5.9), SW - Inches(1.8), Inches(1))
para(b.text_frame, 'CS701 · Team NN · Member A, Member B, Member C, Member D', 16, False, RGBColor(0xB8, 0xC4, 0xD4), first=True)
para(b.text_frame, '11 Nov 2026', 14, False, RGBColor(0x99, 0xA5, 0xB3))
note(s, 'One line intro: one frozen backbone, two heads; the score is driven by detection of objects smaller than a ViT patch.')

# ---------------- Slide 2: the task ----------------
s = slide()
title_bar(s, 'One model, two outputs — on radar images', 'THE TASK')
bullets(s, [
    ('Input: one 8-bit gray SAR image (75% 512×512, 25% 256×256)', 0, False),
    ('Output 1: image class — 9 classes (aircraft … wind_turbine), macro-F1', 0, False),
    ('Output 2: a box + class + score per object (all share the image class), COCO mAP', 0, False),
    ('Ranking: Δm = mean relative gain over ViT-B/16 + LoRA reference', 0, True),
    ('Budget: ≤130 M total / ≤40 M trainable params · one model · no TTA · ≤512 px in', 0, False),
    ('The catch: 52.6% of boxes < 16 px — smaller than one ViT patch (median 15 px)', 0, True),
], l=0.75, t=1.35, w=7.6, size=16)
s.shapes.add_picture(str(FIG / 'sar_example.png'), Inches(8.6), Inches(1.55), height=Inches(5.3))
note(s, 'SAR = microwave echo strength, not a photo: speckle, bright metal, dark water. Δm=0 means "as good as the reference". Right: a wind_turbine image — 29 objects at ~5 px each.')

# ---------------- Slide 3: why obvious fails ----------------
s = slide()
title_bar(s, 'Where the score lives', 'MOTIVATION')
rows = [
    ['Run (val reference)', 'macro-F1', 'mAP', 'Δm'],
    ['ViT + LoRA (reference)', '93.3', '26.1', '0'],
    ['ViT from scratch (over budget)', '83.5', '18.2', '−20.3'],
    ['ViT full FT (over budget)', '94.4', '32.0', '+11.8'],
]
table(s, rows, 0.75, 1.5, 7.6, 1.6, col_w=[3.4, 1.4, 1.4, 1.4], size=15)
bullets(s, [
    ('Pretraining is everything: scratch ≈ −20 Δm', 0, False),
    ('F1 saturates ≈ 94–96 → headroom is in mAP', 0, True),
    ('8 px box + 2 px offset ⇒ IoU ≈ 0 → IoU matching starves tiny objects', 0, False),
    ('SAR ≠ RGB: 1 channel, speckle, own mean/std', 0, False),
], l=8.6, t=1.45, w=4.2, size=15)
note(s, 'Even the over-budget full fine-tune only reaches +11.8: the ceiling is not capacity, it is the detector design for tiny objects.')

# ---------------- Slide 4: architecture ----------------
s = slide()
title_bar(s, 'The model: frozen DINOv3 + LoRA, two heads', 'METHOD')
s.shapes.add_picture(str(FIG / 'model_diagram.png'), Inches(0.75), Inches(1.4), height=Inches(4.9))
mono(s, '''post: score = p_det · obj^0.25 · soft-NMS σ.75 · stage-mean boxes · −0.4 px shift · ≥30 ≤100 boxes
image class = argmax_c  log p_img(c) + log q_det(c)
23.8 M trainable / 109 M total · one forward pass''', 0.85, 6.35, 11.7, 1.0, size=15)
note(s, 'One forward pass per image produces both outputs. Frozen backbone → we only train LoRA + heads, 23.8M of the 40M budget.')

# ---------------- Slide 5: results ----------------
s = slide()
title_bar(s, 'Ablation ladder → +38.23 on the leaderboard', 'RESULTS')
s.shapes.add_picture(str(FIG / 'ablation_ladder.png'), Inches(0.75), Inches(1.45), height=Inches(3.9))
chip(s, 'macro-F1 95.06', 9.35, 1.6, 3.2)
chip(s, 'mAP 45.62', 9.35, 2.3, 3.2, GREEN)
chip(s, 'AP50 70.38', 9.35, 3.0, 3.2, GREEN)
chip(s, 'Δm +38.23', 9.35, 3.7, 3.2, GREEN)
bullets(s, [
    ('Biggest wins: DINOv3 (+8), wide LoRA (+10), tiny-object detector (+13)', 0, False),
    ('mAP drives Δm: +19.5 over ref; F1 +1.75', 0, False),
    ('Beats the over-budget full-FT upper bound (+13.5)', 0, True),
], l=0.75, t=5.5, w=12, size=16)
note(s, 'Screening rule: only differences > ~1.5 Δm trusted (seed noise ±1). Local holdout over-estimates the leaderboard by ~3 mAP.')

# ---------------- Slide 6: post-processing ----------------
s = slide()
title_bar(s, 'Free mAP on frozen outputs (one forward pass)', 'POST-PROCESSING')
rows = [
    ['Step', 'Δ mAP'],
    ['score = p_det × RPN objectness^0.25, per-class soft-NMS σ .75', '+1.3–1.8'],
    ['box = mean of the 3 cascade stages', '+0.3–0.5'],
    ['−0.4 px shift (measured annotation bias)', '+0.4–0.7'],
    ['threshold 1e−5, keep ≥30 ≤100 boxes', '+0.1–0.2'],
    ['image class = argmax log p_img + log q_det', '±0.2 F1'],
]
table(s, rows, 0.75, 1.5, 9.2, 2.7, col_w=[7.2, 2.0], size=14)
bullets(s, [
    ('All steps reuse the same single forward pass — no ensemble, no TTA', 0, False),
    ('Raw outputs dumped once → whole study re-scored in seconds (pp_study.py)', 0, False),
    ('≈ +2–3 mAP total on the same model', 0, True),
], l=0.75, t=4.5, w=12, size=16)
note(s, 'COCO mAP rewards keeping low-scored boxes: every image has ≥1 object so a 30-box floor only adds recall.')

# ---------------- Slide 7: negatives ----------------
s = slide()
title_bar(s, 'What did not work (and what we kept)', 'NEGATIVE RESULTS')
bullets(s, [
    ('token-stride 8 (overlapping patches): +bridge/playground/turbine, −0.4 mAP overall (oil tank) · 2.6× cost', 0, False),
    ('LoRA rank 64 · Mona adapters · multi-level ViT neck · ConvNeXt backbones', 0, False),
    ('D4 rotations + same-class copy-paste augmentation', 0, False),
    ('DINO-YOLO cls-head: +3–4 F1 but −1.5 mAP → kept detector-vote class', 0, False),
    ('', 0, False),
    ('Selection discipline: screen on the stratified local split (80/10/10),', 0, True),
    ('submit to the leaderboard at most what the holdout justifies', 0, True),
], size=17)
note(s, 'Every dropped option has a local-split measurement behind it — this is the "interesting extensions" bonus evidence.')

# ---------------- Slide 8: efficiency + YOLO family ----------------
s = slide()
title_bar(s, 'Budget & a lighter single-shot family', 'EFFICIENCY')
rows = [
    ['Model', 'total', 'trainable', 'local Δm', 'note'],
    ['Ours (V2)', '109.0 M', '23.8 M', '+38.2 (LB)', '15 ms/img, 3.5 h on 2×RTX 6000'],
    ['DINO-YOLO26-s', '31.9 M', '31.9 M', '+21.3', 'single-shot; class from detections'],
    ['improved YOLO11-s', '~9.5 M', '~9.5 M', '+20.5 (LB)', '48 epochs'],
    ['Reference LoRA', '107 M', '21.0 M', '0', 'ViT-B/16 + Faster R-CNN'],
]
table(s, rows, 0.7, 1.5, 12.0, 2.4, col_w=[2.6, 1.5, 1.6, 1.8, 4.5], size=13)
bullets(s, [
    ('DINO features fused into YOLO via zero-init 1×1 convs: +3–4 mAP over plain YOLO26-s', 0, False),
    ('Two-stage wins on mAP (+8–9) at 3× the parameters — accuracy vs footprint trade-off', 0, False),
    ('Where the 40 M budget went: ~20 M heads, ~3.5 M LoRA + norms', 0, False),
], l=0.75, t=4.2, w=12, size=16)
note(s, 'DINO-YOLO: DINOv3 ViT-S features resized + zero-init 1×1 conv added to P3/P4/P5 — training starts from COCO-pretrained YOLO.')

# ---------------- Slide 9: phase 2 + conclusion ----------------
s = slide()
title_bar(s, 'Phase 2 plan & take-away', 'WRAP-UP')
bullets(s, [
    ('Phase 2 (test, 2 submissions only):', 0, True),
    ('F1_full — V2 recipe retrained on all 9,392 labelled images, 100 epochs', 1, False),
    ('Second slot: best-screened variant (EMA / second seed) — raw outputs saved', 1, False),
    ('for post-hoc re-scoring without re-running the model', 1, False),
    ('', 0, False),
    ('Take-away:', 0, True),
    ('Freeze a strong self-supervised backbone; spend the budget on wide LoRA', 1, False),
    ('and a detector rebuilt for sub-patch objects — detection is where Δm lives', 1, False),
], size=17)
note(s, 'Close on the single sentence. Q&A backup slides follow.')

# ---------------- Backup slides ----------------
s = slide()
title_bar(s, 'Backup: per-class & size analysis', 'Q&A')
bullets(s, [
    ('Weak classes: bridge, playground, wind_turbine (confuse with each other & oil_tank)', 0, False),
    ('[per-class AP table from tools/size_ap.py — insert]', 0, False),
    ('[AP by box size — insert]', 0, False),
    ('Local holdout over-estimates Codabench val by ~3 mAP (48.9 vs 45.6)', 0, False),
], size=16)
note(s, 'Fill with tools/size_ap.py output on V2 raw dumps.')

s = slide()
title_bar(s, 'Backup: submission format & reproducibility', 'Q&A')
mono(s, '''submission.zip
├── classification.csv    image_id,label    (one row per image, label 0–8)
└── detection.json        [{"image_id","category_id" 1–9,"bbox" [x,y,w,h] orig px,"score"}]

check_submission.py → format rules + box/class agreement
predict.py --raw-out → raw outputs kept for re-scoring
train.py prints total & trainable parameters (109,014,752 / 23,805,152)''', 0.85, 1.5, 11.7, 3.2, size=14)
note(s, 'Mention seed policy: gains < ~2 reported as mean±std in the report.')

prs.save('CS701_talk_draft.pptx')
print('wrote CS701_talk_draft.pptx with', len(prs.slides.slides if hasattr(prs.slides, "slides") else prs.slides._sldIdLst), 'slides')
