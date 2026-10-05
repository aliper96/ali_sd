"""
gen_ad.py — synthetic pool from the AnomalyDiffusion baseline under OUR protocol (run inside the AD repo, AD venv).

Same inputs as Ali-AUG's gen_syn.py for one category: for every TRAIN-fold sample of split 0 it generates one
image from the sample's own mask (train_A) and clean input (train_C: random good image for textures, the
inpainted defective image for objects), requesting the mask's own defect type. Generation follows the official
generate_with_mask.py: model.log_images(..., inpaint=True, unconditional_only=True), its defaults (DDIM 200 steps,
eta 1, 256 px), adaptive attention re-weighting for texture categories (their README: texture anomalies on,
structural anomalies off). Outputs are resized 256 -> 512 (bicubic) for the shared YOLO pipeline.
Writes <out>/<name>.png, prompts.json (name -> the split's own text prompt, so build_yolo --label-from prompt
labels each image with the requested type) and labels.json; --time also records seconds per image.

    python gen_ad.py --split /mnt/scratch/t23/aliaug_cat/<cat>/split_0 --ckpt-dir logs/<run>/checkpoints \
        --out /mnt/scratch/t23/ad_syn_<cat> [--time results_timing/ad_<cat>.json --limit 20]
"""
import argparse
import json
import os
import random
import sys
import time
import zlib

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.getcwd())
from ldm.data.personalized import imagenet_templates_small  # noqa: E402
from ldm.util import instantiate_from_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--ckpt-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--time", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--fold", default="train")
ap.add_argument("--no-ckpt", action="store_true",
                help="timing / smoke test only: untrained embeddings (same compute); never for downstream data")
a = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvtec_names import TEXTURES, parse  # noqa: E402

config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml")
ckpt = "./models/ldm/text2img-large/model.ckpt"
sd = torch.load(ckpt, map_location="cpu")["state_dict"]
config.model.params.ckpt_path = ckpt
model = instantiate_from_config(config.model)
model.load_state_dict(sd, strict=False)
model = model.cuda()
model.prepare_spatial_encoder(optimze_together=True)
if hasattr(torch.serialization, "add_safe_globals"):  # torch>=2.4: keep weights_only loading, allow
    torch.serialization.add_safe_globals([torch.nn.ParameterDict, torch.nn.Parameter])  # only these classes
if not a.no_ckpt:
    model.embedding_manager.spatial_encoder_model.load_state_dict(
        torch.load(os.path.join(a.ckpt_dir, "spatial_encoder.pt")))
    model.embedding_manager.load(os.path.join(a.ckpt_dir, "embeddings.pt"))
model.eval()

prompts = json.load(open(os.path.join(a.split, f"{a.fold}_prompts.json"), encoding="utf-8"))
names = sorted(prompts)[: a.limit or None]
os.makedirs(a.out, exist_ok=True)
used, labels, times = {}, {}, []
B = 8  # the official generate_with_mask.py uses batch 8; with batch 1 their get_input squeezes the batch dim
torch.cuda.reset_peak_memory_stats()
for b0 in range(0, len(names), B):
    chunk = names[b0:b0 + B]
    real = len(chunk)
    chunk = chunk + [chunk[-1]] * (B - real)  # pad the last batch; padded outputs are discarded
    random.seed(zlib.crc32(chunk[0].encode()))
    torch.manual_seed(zlib.crc32(chunk[0].encode()))
    imgs, masks, keys = [], [], []
    for name in chunk:
        cat, typ = parse(name)
        img = Image.open(os.path.join(a.split, f"{a.fold}_C", name)).convert("RGB").resize((256, 256), Image.BICUBIC)
        m = Image.open(os.path.join(a.split, f"{a.fold}_A", name)).convert("L").resize((256, 256), Image.BICUBIC)
        imgs.append(np.asarray(img, np.float32) / 127.5 - 1.0)
        masks.append((np.asarray(m) > 25).astype(np.float32))  # any non-black pixel (white or colour masks)
        keys.append(f"{cat}+{typ}")
    batch = {"image": torch.from_numpy(np.stack(imgs)).cuda(), "mask": torch.from_numpy(np.stack(masks)).cuda(),
             "name": keys, "caption": [random.choice(imagenet_templates_small).format("*") for _ in chunk]}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), model.ema_scope():
        out = model.log_images(batch, N=B, sample=False, inpaint=True, unconditional_only=True,
                               adaptive_mask=parse(chunk[0])[0] in TEXTURES)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) / B)
    for k, name in enumerate(chunk[:real]):
        x = ((out["samples_inpainting"][k].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
        Image.fromarray((x * 255).round().astype(np.uint8)).resize((512, 512), Image.BICUBIC).save(
            os.path.join(a.out, name))
        cat, typ = parse(name)
        used[name] = prompts[name]
        labels[name] = {"requested": typ, "mask": typ, "category": cat}
json.dump(used, open(os.path.join(a.out, "prompts.json"), "w", encoding="utf-8"), indent=1)
json.dump(labels, open(os.path.join(a.out, "labels.json"), "w", encoding="utf-8"), indent=1)
print(f"generated {len(used)} images -> {a.out}")
if a.time:
    t = np.array(times[1:] if len(times) > 2 else times)  # the first batch includes CUDA warm-up
    res = {"method": "AnomalyDiffusion (official code, DDIM 200)", "untrained_embeddings": a.no_ckpt,
           "resolution": 256, "batch": B, "steps": 200, "n_batches": int(len(t)),
           "s_per_img_mean": float(t.mean()), "s_per_img_std": float(t.std()),
           "s_per_img_median": float(np.median(t)), "img_per_s": float(1 / t.mean()),
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
           "params_total_M": sum(p.numel() for p in model.parameters()) / 1e6,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    os.makedirs(os.path.dirname(os.path.abspath(a.time)), exist_ok=True)
    json.dump(res, open(a.time, "w"), indent=1)
    print(json.dumps(res, indent=1))
