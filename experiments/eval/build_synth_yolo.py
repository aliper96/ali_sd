"""
build_synth_yolo.py — Turn Ali-AUG generations into a YOLO-seg synthetic dataset.

The downstream CAS/NAS evaluation (run_downstream_yolo.py) needs SYNTHETIC images
with labels in the SAME format as the real set (results/data/real_tile). This
builder is the glue between the generator's output and that eval: it takes a flat
folder of generated images named by their source sample-id and, for each, derives

  * the segmentation polygon from the MASK that conditioned the generation
    (dataset/mvtec_defects/mvtec/inputs/<sid>/mask.*), and
  * the class from the defect token in the sample-id
    "<category>__<defect>__<stem>"  ->  class = <defect>  (the prompt-assigned,
    i.e. weak, label; a failed generation can therefore carry a wrong label — the
    exact qualification stated in the paper's limitations).

Inference-time only: it consumes images the generator already produced. It does
NOT train anything. Generate the images first (sample the trained checkpoint over
the real defect masks) into --gen-dir, then run this.

Why not convert_mvtec_to_yolo.py --mode synthetic? That mode expects per-class
subfolders with a sibling <stem>_mask.png, or a flat dir + a separate --mask-dir.
Our generations are named <sid>_step..._cfg....png with NO mask saved next to them,
and the true conditioning mask lives in the dataset keyed by sid. This script
resolves that mapping. It reuses mask_to_polygons / write_label from the converter
so the label format is identical.

Output layout (feed to run_downstream_yolo.py --syn-images-dir/--syn-labels-dir):
    <out>/images/<sid>.png
    <out>/labels/<sid>.txt        # "<cls> x1 y1 x2 y2 ..." normalised polygon

Usage:
    python eval/build_synth_yolo.py \\
        --gen-dir samples_all_categories \\
        --category tile --names crack glue_strip gray_stroke oil \\
        --out results/data/syn_tile

Dependencies:
    pip install opencv-python numpy   (via convert_mvtec_to_yolo.mask_to_polygons)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from convert_mvtec_to_yolo import mask_to_polygons, write_label, _link_or_copy  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MASK_STEMS = ("mask",)  # dataset stores the conditioning mask as inputs/<sid>/mask.*

# strip generation suffixes to recover the bare sample-id:
#   tile__crack__000_step28_cfg3.0(.png) -> tile__crack__000
_SUFFIX_RE = re.compile(r"_step\d+.*$")


def _gen_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def sid_of(stem: str) -> str:
    """Recover the dataset sample-id from a generated file stem."""
    return _SUFFIX_RE.sub("", stem)


def find_mask(inputs_root: Path, sid: str) -> Optional[Path]:
    sdir = inputs_root / sid
    if not sdir.is_dir():
        return None
    for stem in MASK_STEMS:
        for ext in IMAGE_EXTS:
            p = sdir / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def build(gen_dir: Path, inputs_root: Path, names: List[str],
          category: Optional[str], out: Path, min_area: int) -> None:
    name_to_id = {n: i for i, n in enumerate(names)}
    kept = {n: 0 for n in names}
    skipped_maskonly = skipped_class = skipped_nomask = skipped_nopoly = 0

    for img in _gen_images(gen_dir):
        stem = img.stem
        # skip demo artefacts and the mask-only-mode variant (different image, same mask)
        if stem.startswith("montage") or "_maskonly" in stem:
            skipped_maskonly += 1
            continue

        sid = sid_of(stem)
        parts = sid.split("__")
        if len(parts) < 2:
            print(f"[skip] {img.name}: sid '{sid}' has no '<cat>__<defect>__...' form")
            skipped_class += 1
            continue
        cat, defect = parts[0], parts[1]
        if category and cat != category:
            continue
        if defect not in name_to_id:
            print(f"[skip] {img.name}: defect '{defect}' not in --names {names}")
            skipped_class += 1
            continue

        mask = find_mask(inputs_root, sid)
        if mask is None:
            print(f"[skip] {img.name}: no conditioning mask at {inputs_root / sid}/mask.*")
            skipped_nomask += 1
            continue

        polys = mask_to_polygons(mask, min_area=min_area)
        if not polys:
            print(f"[skip] {img.name}: mask produced no polygon (empty/too small)")
            skipped_nopoly += 1
            continue

        # Output keyed by the FULL generation stem (unique across seeds/cfg), not
        # the sid — otherwise several seeds of the same sample would overwrite each
        # other. sid is used only to look up the shared mask/class.
        _link_or_copy(img, out / "images" / f"{stem}{img.suffix}")
        write_label(out / "labels" / f"{stem}.txt", name_to_id[defect], polys)
        kept[defect] += 1

    total = sum(kept.values())
    print("\nSynthetic -> YOLO-seg summary")
    print(f"  kept {total} images  {kept}")
    print(f"  skipped: maskonly/montage={skipped_maskonly} class={skipped_class} "
          f"no_mask={skipped_nomask} no_polygon={skipped_nopoly}")
    print(f"  output: {out.resolve()}  (images/, labels/)")
    if total == 0:
        print("  WARNING: 0 images kept — generate synthetics into --gen-dir first "
              "(sample the checkpoint over the real defect masks).")
    else:
        print("  NOTE: labels use the prompt-assigned (weak) class; a failed generation "
              "may carry a wrong label (stated in the paper's limitations).")


def _cli():
    p = argparse.ArgumentParser(description="Ali-AUG generations -> YOLO-seg synthetic dataset.")
    p.add_argument("--gen-dir", required=True, help="Flat dir of generated images named by sample-id.")
    p.add_argument("--inputs-root", default=None,
                   help="dataset/mvtec_defects/mvtec/inputs (source of conditioning masks). "
                        "Default: <repo>/dataset/mvtec_defects/mvtec/inputs")
    p.add_argument("--names", nargs="+", default=["crack", "glue_strip", "gray_stroke", "oil"],
                   help="Class order; must match the real set's --names.")
    p.add_argument("--category", default=None,
                   help="Keep only this MVTec category (e.g. tile). Default: all categories.")
    p.add_argument("--out", required=True, help="Output dataset root (creates images/ and labels/).")
    p.add_argument("--min-area", type=int, default=20, help="Drop mask components smaller than this (px).")
    args = p.parse_args()

    repo_root = ROOT.parent
    inputs_root = Path(args.inputs_root) if args.inputs_root else \
        repo_root / "dataset" / "mvtec_defects" / "mvtec" / "inputs"
    if not inputs_root.is_dir():
        p.error(f"inputs-root not found: {inputs_root}")

    build(Path(args.gen_dir), inputs_root, args.names, args.category,
          Path(args.out), args.min_area)


if __name__ == "__main__":
    _cli()
