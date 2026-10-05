"""
judge.py — E2: label fidelity of synthetic defects, measured by an independent classifier.

Answers review theme B ("a prompt is not a verified label") with a number instead of a caveat.
For one split, and separately for every category present in the synthetic set:
  * crops around each defect (mask bounding box + margin, 224 px) — the classifier sees the
    defect, not the whole image;
  * judge = ImageNet EfficientNet-B0 fine-tuned ONLY on the REAL training crops of that category,
    to tell apart its defect types (never on synthetic images, never on the test fold);
  * CEILING = judge accuracy on the REAL held-out test crops (what a perfect generator could get);
  * FIDELITY = judge accuracy on the SYNTHETIC crops, scored against the type each synthetic image
    was generated to carry (labels.json from gen_syn.py; prompt or colour);
  * FOLLOWS_MASK = how often the judge sees the type of the mask SHAPE (differs only in swap runs);
  * CHANCE = 1 / number of types of the category.
Overall numbers are sample-weighted over categories; per-category numbers are kept.
With a single category (tile) this is exactly the tile-only judge used before.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvtec_names import parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, help="white-mask split (geometry + real images)")
ap.add_argument("--syn", required=True, help="synthetic dir with prompts.json (+ labels.json)")
ap.add_argument("--label-from", choices=["prompt", "name"], default="prompt")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--fold", choices=["train", "test"], default="train",
                help="fold whose masks generated the synthetics (for the crop geometry)")
a = ap.parse_args()
split, syn = Path(a.split), Path(a.syn)


def crop(img_path, mask_path, size=224, margin=0.25):
    img = Image.open(img_path).convert("RGB")
    m = np.asarray(Image.open(mask_path).convert("L").resize(img.size)) > 127
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return img.resize((size, size))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    s = int(max(y1 - y0 + 1, x1 - x0 + 1) * (1 + 2 * margin)) + 16
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return img.crop((cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2)).resize((size, size))


# (category, prompt) -> type, for synthetic sets without labels.json
p2t = {}
for fold in ("train", "test"):
    for n, p in json.load(open(split / f"{fold}_prompts.json", encoding="utf-8")).items():
        p2t[(parse(n)[0], p)] = parse(n)[1]

used = json.load(open(syn / "prompts.json", encoding="utf-8"))
lab_file = syn / "labels.json"
lab = json.load(open(lab_file, encoding="utf-8")) if lab_file.exists() else {}
syn_by_cat = {}
for n, p in sorted(used.items()):
    mname = n.split("__")[0] + ".png" if "__" in n else n
    cat, t_mask = parse(mname)
    req = lab[n]["requested"] if n in lab else (p2t[(cat, p)] if a.label_from == "prompt" else t_mask)
    syn_by_cat.setdefault(cat, []).append((n, mname, req, t_mask))

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
aug = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                 T.RandomVerticalFlip(), T.RandomRotation(90), T.ColorJitter(0.2, 0.2, 0.1),
                 T.ToTensor(), norm])
plain = T.Compose([T.ToTensor(), norm])
import timm  # noqa: E402

per_cat, disagreements = {}, []
tot = dict(n_syn=0, ok=0, mask=0, n_test=0, test_ok=0, chance_w=0.0)
for cat, items in sorted(syn_by_cat.items()):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    real = {f: [(n.name, parse(n.name)[1]) for n in sorted((split / f"{f}_B").iterdir())
                if parse(n.name)[0] == cat] for f in ("train", "test")}
    types = sorted({t for f in real for _, t in real[f]})
    if len(types) < 2:
        continue
    X = [(crop(split / "train_B" / n, split / "train_A" / n), types.index(t)) for n, t in real["train"]]
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(types)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(a.epochs):
        model.train()
        idx = np.random.permutation(len(X))
        for i in range(0, len(idx), 16):
            b = [X[j] for j in idx[i:i + 16]]
            x = torch.stack([aug(im) for im, _ in b]).cuda()
            y = torch.tensor([c for _, c in b]).cuda()
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = lambda im: types[int(model(plain(im).unsqueeze(0).cuda()).argmax(1))]  # noqa: E731
        test_ok = [pred(crop(split / "test_B" / n, split / "test_A" / n)) == t for n, t in real["test"]]
        res = [(n, req, tm, pred(crop(syn / n, split / f"{a.fold}_A" / mname)))
               for n, mname, req, tm in items]
    ok = [j == r for _, r, _, j in res]
    fm = [j == tm for _, _, tm, j in res]
    per_cat[cat] = dict(types=types, n_syn=len(res), n_test=len(test_ok), chance=1 / len(types),
                        ceiling=float(np.mean(test_ok)), fidelity=float(np.mean(ok)),
                        follows_mask=float(np.mean(fm)))
    disagreements += [dict(name=n, category=cat, requested=r, judged=j) for n, r, _, j in res if j != r]
    tot["n_syn"] += len(res); tot["ok"] += sum(ok); tot["mask"] += sum(fm)
    tot["n_test"] += len(test_ok); tot["test_ok"] += sum(test_ok); tot["chance_w"] += len(res) / len(types)
    print(f"  {cat:12s} ceiling {per_cat[cat]['ceiling']:.3f}  fidelity {per_cat[cat]['fidelity']:.3f}  "
          f"follows_mask {per_cat[cat]['follows_mask']:.3f}  (n={len(res)}, {len(types)} types)", flush=True)

out = dict(
    split=str(split), syn=str(syn), label_from=a.label_from, seed=a.seed, fold=a.fold,
    n_syn=tot["n_syn"], n_test_real=tot["n_test"],
    chance=tot["chance_w"] / max(tot["n_syn"], 1),
    ceiling=tot["test_ok"] / max(tot["n_test"], 1),
    fidelity=tot["ok"] / max(tot["n_syn"], 1),
    follows_mask=tot["mask"] / max(tot["n_syn"], 1),
    per_category=per_cat, disagreements=disagreements,
)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"ceiling {out['ceiling']:.3f}  fidelity {out['fidelity']:.3f}  follows_mask {out['follows_mask']:.3f}  "
      f"chance {out['chance']:.3f}  (n_syn={out['n_syn']})")
