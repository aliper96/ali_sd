"""Compose the failure-case figure (review theme Q) from existing unpaired
generations. No GPU. Two panels:
  A) weak/failed defect insertion on small-mask categories (input|mask|gen|real)
  B) abandoned mask-only mode: category-averaging mush (mask|gen)
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r"C:\Users\aliha\Desktop\PHD\FIRSTPAPER-FINAL"
INP = os.path.join(ROOT, "dataset", "mvtec_defects", "mvtec", "inputs")
OUT = os.path.join(ROOT, "dataset", "mvtec_defects", "mvtec", "outputs")
GEN = os.path.join(ROOT, "samples_all_categories")
DST = os.path.join(ROOT, "revision_assets")
os.makedirs(DST, exist_ok=True)

CELL = 224
PAD = 6
LABELH = 22
TITLEH = 30
BG = (245, 245, 245)
FG = (20, 20, 20)

def font(sz):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()

F_LBL = font(13)
F_TIT = font(16)
F_CAP = font(12)

def load(p):
    return Image.open(p).convert("RGB").resize((CELL, CELL)) if os.path.isfile(p) else \
           Image.new("RGB", (CELL, CELL), (60, 60, 60))

# Panel A: (sample_id, gen_filename, failure note)
PANEL_A = [
    ("capsule__crack__000",            "capsule__crack__000_step28_cfg3.0.png",            "defect barely inserted (tiny mask)"),
    ("carpet__color__000",             "carpet__color__000_step28_cfg3.0.png",             "region altered but defect weak"),
    ("leather__color__000",            "leather__color__000_step28_cfg3.0.png",            "prompt-assigned label unreliable"),
    ("screw__manipulated_front__000",  "screw__manipulated_front__000_step28_cfg3.0.png",  "defect absent -> wrong weak label"),
    ("wood__color__000",               "wood__color__000_step28_cfg3.0.png",               "object preserved, no visible defect"),
]

# Panel B: the REAL mask-only failure from the dedicated mask-only model (step
# 3000). Honest example: prompt='metal nut' + a bottle-shaped mask -> incoherent
# blend of categories (brown granular + mesh patch), no reference anchor.
MASKONLY_MASK = os.path.join(INP, "bottle__broken_large__000", "mask.jpg")
MASKONLY_GEN  = os.path.join(ROOT, "checkpoints", "mvtec_defects_maskonly",
                             "samples", "step_003000", "00.png")

def col_headers(cols, width):
    band = Image.new("RGB", (width, LABELH), (225, 225, 225))
    d = ImageDraw.Draw(band)
    n = len(cols)
    for i, c in enumerate(cols):
        d.text((i * (CELL + PAD) + 4, 4), c, fill=FG, font=F_LBL)
    return band

def title_band(txt, width):
    band = Image.new("RGB", (width, TITLEH), (30, 30, 30))
    ImageDraw.Draw(band).text((6, 7), txt, fill=(255, 255, 255), font=F_TIT)
    return band

def build_panel_A():
    cols = ["clean input", "mask", "generated", "real target", ""]
    ncol = 4
    row_w = ncol * CELL + (ncol - 1) * PAD
    note_w = 240
    width = row_w + PAD + note_w
    rows = []
    for sid, genf, note in PANEL_A:
        cells = [
            load(os.path.join(INP, sid, "image_0.png")),
            load(os.path.join(INP, sid, "mask.jpg")),
            load(os.path.join(GEN, genf)),
            load(os.path.join(OUT, sid, "image.png")),
        ]
        row = Image.new("RGB", (width, CELL), BG)
        for i, c in enumerate(cells):
            row.paste(c, (i * (CELL + PAD), 0))
        d = ImageDraw.Draw(row)
        cat = sid.split("__")[0]
        d.text((row_w + PAD + 6, 8), cat, fill=FG, font=F_TIT)
        # wrap note
        words, line, y = note.split(), "", 40
        for w in words:
            if len(line + " " + w) > 28:
                d.text((row_w + PAD + 6, y), line, fill=(120, 20, 20), font=F_CAP); y += 16; line = w
            else:
                line = (line + " " + w).strip()
        d.text((row_w + PAD + 6, y), line, fill=(120, 20, 20), font=F_CAP)
        rows.append(row)
    hdr = col_headers(cols, width)
    title = title_band("Weak / failed defect insertion on small-mask categories (unpaired mode)", width)
    H = title.height + hdr.height + sum(r.height + PAD for r in rows)
    panel = Image.new("RGB", (width, H), BG)
    panel.paste(title, (0, 0)); y = title.height
    panel.paste(hdr, (0, y)); y += hdr.height
    for r in rows:
        panel.paste(r, (0, y)); y += r.height + PAD
    return panel

def build_panel_B():
    ncol = 2
    row_w = ncol * CELL + (ncol - 1) * PAD
    note_w = 360
    width = row_w + PAD + note_w
    cells = [load(MASKONLY_MASK), load(MASKONLY_GEN)]
    row = Image.new("RGB", (width, CELL), BG)
    for i, c in enumerate(cells):
        row.paste(c, (i * (CELL + PAD), 0))
    d = ImageDraw.Draw(row)
    note = ('prompt = "metal nut" + this mask, no reference image. '
            'Output is an incoherent blend of categories (granular + mesh) '
            'with no coherent object -> the mask-only model averages the 15 '
            'categories. Mode abandoned.')
    words, line, y = note.split(), "", 8
    for w in words:
        if len(line + " " + w) > 44:
            d.text((row_w + PAD + 6, y), line, fill=(120, 20, 20), font=F_CAP); y += 16; line = w
        else:
            line = (line + " " + w).strip()
    d.text((row_w + PAD + 6, y), line, fill=(120, 20, 20), font=F_CAP)
    hdr = col_headers(["mask", "generated (no reference)"], width)
    title = title_band("B. Abandoned mask-only mode (dedicated model, step 3000): category averaging", width)
    H = title.height + hdr.height + row.height + PAD
    panel = Image.new("RGB", (width, H), BG)
    panel.paste(title, (0, 0)); y = title.height
    panel.paste(hdr, (0, y)); y += hdr.height
    panel.paste(row, (0, y))
    return panel

# Panel B (mask-only) intentionally omitted: mask-only is being improved, not
# reported as a failure. Only Panel A (weak defect insertion on small-mask
# categories, the reported unpaired mode) is kept.
A = build_panel_A()
out = os.path.join(DST, "fig_failure_cases.png")
A.save(out)
print("saved:", out, A.size)
