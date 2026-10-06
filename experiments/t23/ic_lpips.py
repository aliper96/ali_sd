"""ic_lpips.py — intra-class LPIPS diversity (IC-LPIPS, as in AnomalyDiffusion) of the held-out synthetic
sets, per category: mean pairwise LPIPS (AlexNet, 256 px) between images of the same defect type, averaged
over types, for AnomalyDiffusion, Ali-AUG text and Ali-AUG colour (review 2026-10-05: no diversity metric).
  python prep/t23/ic_lpips.py --cat-root D:/ad/aliaug_cat --ad D:/ad/judge_syn --ours D:/ad/ours_heldout --out <live>/ic_lpips.json
"""
import argparse, itertools, json, os, sys
import numpy as np, torch, lpips
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvtec_names import parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cat-root", required=True)
ap.add_argument("--ad", required=True)
ap.add_argument("--ours", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-pairs", type=int, default=200)
a = ap.parse_args()
net = lpips.LPIPS(net="alex").cuda().eval()


def load(p):
    x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS))).permute(2, 0, 1).float() / 127.5 - 1
    return x.unsqueeze(0).cuda()


res = {}
cats = sorted(d for d in os.listdir(a.cat_root) if os.path.isdir(os.path.join(a.cat_root, d)))
for cat in cats:
    gens = {"AnomalyDiffusion": os.path.join(a.ad, f"ad_heldout_{cat}"),
            "Ali-AUG text": os.path.join(a.ours, f"cattext_heldout_{cat}"),
            "Ali-AUG colour": os.path.join(a.ours, f"catcolor_heldout_{cat}")}
    res[cat] = {}
    for g, d in gens.items():
        if not os.path.isdir(d):
            continue
        by_type = {}
        for n in sorted(os.listdir(d)):
            if n.endswith(".png"):
                by_type.setdefault(parse(n)[1], []).append(os.path.join(d, n))
        vals = []
        rng = np.random.RandomState(0)
        for t, files in by_type.items():
            pairs = list(itertools.combinations(files, 2))
            if len(pairs) < 1:
                continue
            if len(pairs) > a.max_pairs:
                pairs = [pairs[i] for i in rng.choice(len(pairs), a.max_pairs, replace=False)]
            with torch.no_grad():
                dv = [float(net(load(x), load(y))) for x, y in pairs]
            vals.append(float(np.mean(dv)))
        res[cat][g] = float(np.mean(vals)) if vals else None
    print(cat, {k: (round(v, 3) if v is not None else None) for k, v in res[cat].items()})
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
