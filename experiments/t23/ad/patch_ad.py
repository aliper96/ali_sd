"""
patch_ad.py — the only changes made to the official AnomalyDiffusion code. Idempotent.
 1. With AD_ALL=1 the training loader uses every file of <cat>/test/<type> instead of the lowest-ID third.
    Our folder (build_ad_data.py) already contains only our train fold, so this keeps their recipe and
    removes their split.
 2. With AD_NATIVE_OPS=1 the StyleGAN2 ops of the psp spatial encoder use pure-PyTorch formulas
    (native_ops.py) instead of JIT-compiled CUDA kernels (no ninja/nvcc needed). Same maths.
    Used for every AD run (training, generation, timing) so all are identical.
"""
import shutil
import sys
from pathlib import Path

repo = Path(sys.argv[1])
f = repo / "ldm" / "data" / "personalized.py"
s = f.read_text(encoding="utf-8")
old = "if set=='train' and idx>len(img_files)//3:"
new = "if set=='train' and idx>len(img_files)//3 and not os.environ.get('AD_ALL'):"
if new not in s:
    assert s.count(old) == 1, s.count(old)
    f.write_text(s.replace(old, new), encoding="utf-8")
print("split patch:", "ok" if new in f.read_text(encoding="utf-8") else "FAILED")

op = repo / "ldm" / "models" / "psp_encoder" / "stylegan2" / "op"
shutil.copy(Path(__file__).with_name("native_ops.py"), op / "native.py")
init = op / "__init__.py"
s = init.read_text(encoding="utf-8")
if "AD_NATIVE_OPS" not in s:
    init.write_text("import os\nif os.environ.get('AD_NATIVE_OPS'):\n"
                    "    from .native import FusedLeakyReLU, fused_leaky_relu, upfirdn2d\nelse:\n"
                    + "".join("    " + line + "\n" for line in s.strip().splitlines()), encoding="utf-8")
print("ops patch:", "ok" if "AD_NATIVE_OPS" in init.read_text(encoding="utf-8") else "FAILED")
