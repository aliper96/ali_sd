"""
build_color_splits.py — the SAME splits with the class carried by the mask COLOUR (or by nothing).

For every split_k of --src it writes --dst/split_k where:
  * {train,test}_A  : the same mask geometry, painted with the colour of its defect type
                      (mvtec_names.colour: index of the type within its category; tile keeps the
                      palette of the tile-only runs). With --white the mask stays white: NO label
                      channel at all (ablation "no label carrier");
  * {train,test}_B/C: symlinks to the originals (identical images, no disk cost);
  * {train,test}_prompts.json : one constant prompt, "Add a defect.", for every sample, so the colour
    is the ONLY carrier of the class (with --white: there is none).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvtec_names import colour, parse, types_by_category  # noqa: E402

PROMPT = "Add a defect."

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--white", action="store_true", help="ablation: white masks, no label carrier")
ap.add_argument("--copy", action="store_true", help="copy B/C instead of symlinking (Windows)")
a = ap.parse_args()

for sp in sorted(Path(a.src).glob("split_*")):
    out = Path(a.dst) / sp.name
    names = [p.name for f in ("train_A", "test_A") for p in (sp / f).iterdir()]
    types_of = types_by_category(names)
    for fold in ("train", "test"):
        for sub in ("B", "C"):
            link = out / f"{fold}_{sub}"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists():
                continue
            if a.copy:
                import shutil
                shutil.copytree(sp / f"{fold}_{sub}", link)
            else:
                os.symlink((sp / f"{fold}_{sub}").resolve(), link)
        dA = out / f"{fold}_A"
        dA.mkdir(parents=True, exist_ok=True)
        prompts = {}
        for f in sorted((sp / f"{fold}_A").iterdir()):
            cat, t = parse(f.name)
            m = np.asarray(Image.open(f).convert("L")) > 127
            rgb = np.zeros((*m.shape, 3), np.uint8)
            rgb[m] = (255, 255, 255) if a.white else colour(cat, t, types_of[cat])
            Image.fromarray(rgb).save(dA / f.name)
            prompts[f.name] = PROMPT
        json.dump(prompts, open(out / f"{fold}_prompts.json", "w"), indent=1)
    print(sp.name, "->", out)
