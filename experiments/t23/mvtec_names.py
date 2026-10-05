"""
mvtec_names.py — sample names, classes and label colours shared by the t23 scripts.

Sample files are named  <category>_<defect type>_<NNN>.png  (build_aliaug_splits.py). Categories and
types can contain underscores (metal_nut, glue_strip), so names are parsed against the known category
list, not split naively.

Class = (category, type). In the COLOUR variant the class is painted into the mask with a colour that
depends on the type's index WITHIN its category (the category itself is given by the input image), so
one palette of 8 colours covers all 73 MVTec-AD types. For tile the index order reproduces the colours
of the tile-only experiments exactly.
"""
import re

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
              "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}  # any defect-free image is pose-compatible

# tile keeps the colours used in every tile experiment (build_color_splits.py / mascara_color.py)
TILE_COLOURS = {"glue_strip": (220, 30, 30), "gray_stroke": (30, 200, 60), "oil": (40, 70, 230),
                "rough": (210, 40, 210), "crack": (240, 170, 20)}
PALETTE = [(220, 30, 30), (30, 200, 60), (40, 70, 230), (210, 40, 210), (240, 170, 20),
           (30, 200, 220), (250, 250, 250), (140, 90, 30)]


def parse(name):
    """'metal_nut_color_003.png' -> ('metal_nut', 'color')."""
    stem = name[:-4] if name.endswith(".png") else name
    stem = stem.split("__")[0]  # extra pools: <stem>__c<j>
    for c in sorted(CATEGORIES, key=len, reverse=True):
        if stem.startswith(c + "_"):
            m = re.match(rf"{re.escape(c)}_(.+)_\d+$", stem)
            if m:
                return c, m.group(1)
    raise ValueError(f"cannot parse sample name {name!r}")


def classes_from(names):
    """Sorted list of 'category/type' classes present in a list of sample names."""
    return sorted({"/".join(parse(n)) for n in names})


def types_by_category(names):
    out = {}
    for n in names:
        c, t = parse(n)
        out.setdefault(c, set()).add(t)
    return {c: sorted(ts) for c, ts in out.items()}


def colour(cat, typ, types_of_cat):
    if cat == "tile" and typ in TILE_COLOURS:
        return TILE_COLOURS[typ]
    return PALETTE[types_of_cat.index(typ) % len(PALETTE)]


def next_type(cat, typ, types_of_cat):
    """Swap test: the next defect type of the SAME category (cyclic)."""
    i = types_of_cat.index(typ)
    return types_of_cat[(i + 1) % len(types_of_cat)]
