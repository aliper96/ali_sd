"""
build_yolo.py — YOLO-seg datasets for ONE split of the Tables 2-3 re-run.

    <out>/<task>/real_train/{images,labels}   train_B (real defective) + polygon from train_A
    <out>/<task>/real_test/{images,labels}    test_B + test_A   (held out; never generated from)
    <out>/<task>/syn/{images,labels}          generated images + polygon from the train_A mask
                                              that conditioned them
with <task> in {binary, multiclass}.

Class of a REAL image: the defect type in its file name (tile_<type>_NNN.png).
Class of a SYNTHETIC image: the defect named by the PROMPT used to generate it — the
prompt-assigned (weak) label of the paper, which a failed generation can get wrong.
The prompt -> type map is derived from the training prompts and checked to be 1:1.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from convert_mvtec_to_yolo import mask_to_polygons, write_label  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mvtec_names import classes_from, parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--syn", required=True, help="gen_syn.py output dir")
ap.add_argument("--out", required=True)
ap.add_argument("--label-from", choices=["prompt", "name"], default="prompt",
                help="class of a synthetic image: the prompt used (text variant) or the "
                     "defect type of the mask it was generated from (colour variant, where "
                     "the requested class is the mask colour and the prompt is constant)")
ap.add_argument("--poly-split", default=None,
                help="split whose *_A masks give the polygons (default --split). For the colour "
                     "variant pass the original white-mask split: same geometry, and the "
                     "grey-threshold polygon extraction needs white masks")
ap.add_argument("--syn-masks", default=None,
                help="SAM step: directory of masks for the SYNTHETIC images (same file names); replaces the "
                     "requested mask as the label geometry")
ap.add_argument("--drop-missing", action="store_true",
                help="with --syn-masks: synthetic images without a mask file (rejected by the SAM step) are dropped")
a = ap.parse_args()
split, syn, out = Path(a.split), Path(a.syn), Path(a.out)
poly = Path(a.poly_split) if a.poly_split else split


# classes = "category/type" present in the split (tile: 5, all MVTec-AD: 73); for tile the index order
# is the same as in the first tile-only runs (sorted type names)
CLASSES = classes_from([p.name for fold in ("train_A", "test_A") for p in (split / fold).iterdir()])


def cls_of(cat, typ):
    return CLASSES.index(f"{cat}/{typ}")


# text variant without labels.json: (category, prompt) -> type, checked 1:1 within each category
p2t = {}
if a.label_from == "prompt":
    for fold in ("train", "test"):
        for n, pmt in json.load(open(split / f"{fold}_prompts.json", encoding="utf-8")).items():
            c, t = parse(n)
            assert p2t.setdefault((c, pmt), t) == t, f"prompt {pmt!r} maps to two types in {c}"


def put(task, part, img, mask, cls):
    d = out / task / part
    (d / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, d / "images" / img.name)
    write_label(d / "labels" / (img.stem + ".txt"), cls, mask_to_polygons(mask))


counts = {}
for part, fold in (("real_train", "train"), ("real_test", "test")):
    for img in sorted((split / f"{fold}_B").iterdir()):
        mask = poly / f"{fold}_A" / img.name
        put("binary", part, img, mask, 0)
        put("multiclass", part, img, mask, cls_of(*parse(img.name)))
        counts[part] = counts.get(part, 0) + 1

used = json.load(open(syn / "prompts.json", encoding="utf-8"))
labf = syn / "labels.json"  # written by gen_syn.py: requested type (+ category)
lab = json.load(open(labf, encoding="utf-8")) if labf.exists() else {}
for name, prompt in sorted(used.items()):
    # extra pools (E3) are named "<mask stem>__c<j>.png": same mask, another clean image
    mname = name.split("__")[0] + ".png" if "__" in name else name
    img, mask = syn / name, poly / "train_A" / mname
    if a.syn_masks:
        sm = Path(a.syn_masks) / name
        if sm.exists():
            mask = sm
        elif a.drop_missing:
            counts["syn_dropped"] = counts.get("syn_dropped", 0) + 1
            continue
    cat, t_mask = parse(mname)
    if name in lab:
        typ = lab[name]["requested"]
    else:
        typ = p2t[(cat, prompt)] if a.label_from == "prompt" else t_mask
    put("binary", "syn", img, mask, 0)
    put("multiclass", "syn", img, mask, cls_of(cat, typ))
    counts["syn"] = counts.get("syn", 0) + 1

(out / "names_binary.txt").write_text("defect\n")
(out / "names_multiclass.txt").write_text("\n".join(c.replace("/", "__") for c in CLASSES) + "\n")
json.dump({"counts": counts, "classes": CLASSES}, open(out / "manifest.json", "w"), indent=1)
print(counts, len(CLASSES), "classes")
