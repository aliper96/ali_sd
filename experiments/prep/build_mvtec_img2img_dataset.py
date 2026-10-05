"""build_mvtec_img2img_dataset.py — convert the MVTec LoRA dataset into the
Img2Img-turbo (train.py / PairedDataset) A/B/C layout.

PairedDataset (utils.py) expects, per split, flat folders keyed by the SAME
filename plus a prompts json:
    <root>/train_A/<name>.png   conditioning = the defect MASK
    <root>/train_B/<name>.png   target       = the DEFECTIVE image
    <root>/train_C/<name>.png   real         = the CLEAN image (no defect)
    <root>/train_prompts.json   {name: prompt}
    (same for test_*)

Source = the existing paired MVTec dataset built by prepare_mvtec_lora_dataset.py:
    dataset/mvtec_defects/mvtec/inputs/<sample>/{image_0.png, mask.jpg, prompt.txt}
    dataset/mvtec_defects/mvtec/outputs/<sample>/image.png

Mapping (from PairedDataset.__getitem__):
    A(mask)  <- inputs/<sample>/mask.*
    B(defect)<- outputs/<sample>/image.*
    C(clean) <- inputs/<sample>/image_0.*
    prompt   <- inputs/<sample>/prompt.txt

Usage (no GPU):
    python prep/build_mvtec_img2img_dataset.py \
        --src dataset/mvtec_defects/mvtec \
        --out dataset/mvtec_img2img \
        --test-frac 0.1
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _find(sample_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = sample_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _save_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(dst, "PNG")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/mvtec_defects/mvtec",
                    help="dir with inputs/ and outputs/")
    ap.add_argument("--out", default="dataset/mvtec_img2img",
                    help="output A/B/C dataset root")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    inputs_root, outputs_root = src / "inputs", src / "outputs"
    if not inputs_root.is_dir() or not outputs_root.is_dir():
        raise SystemExit(f"Expected {inputs_root} and {outputs_root} to exist.")

    samples = sorted(d.name for d in inputs_root.iterdir() if d.is_dir())
    # Deterministic train/test split (stable across runs).
    import random
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_test = max(1, int(len(samples) * args.test_frac))
    split_of = {s: ("test" if i < n_test else "train") for i, s in enumerate(samples)}

    out = Path(args.out)
    prompts: dict[str, dict[str, str]] = {"train": {}, "test": {}}
    kept = {"train": 0, "test": 0}
    skipped = 0

    for s in samples:
        in_dir = inputs_root / s
        mask = _find(in_dir, "mask")
        clean = _find(in_dir, "image_0")
        defect = _find(outputs_root / s, "image")
        prompt_p = in_dir / "prompt.txt"
        if not (mask and clean and defect and prompt_p.exists()):
            skipped += 1
            continue
        split = split_of[s]
        name = f"{s}.png"
        _save_png(mask, out / f"{split}_A" / name)
        _save_png(defect, out / f"{split}_B" / name)
        _save_png(clean, out / f"{split}_C" / name)
        prompts[split][name] = prompt_p.read_text(encoding="utf-8").strip()
        kept[split] += 1

    for split in ("train", "test"):
        (out / f"{split}_prompts.json").write_text(
            json.dumps(prompts[split], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done. train={kept['train']}  test={kept['test']}  skipped={skipped}")
    print(f"Output: {out.resolve()}")
    print("Folders:", ", ".join(sorted(os.listdir(out))))


if __name__ == "__main__":
    main()
