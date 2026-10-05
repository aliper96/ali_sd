"""diag_text.py — where is the text prompt lost?  (tokenizer -> text encoder -> U-Net cross-attn)"""
import importlib.machinery
import sys
import types

try:
    import wandb  # noqa: F401
except ImportError:
    w = types.ModuleType("wandb"); w.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
    sys.modules["wandb"] = w
import os
import torch

code = sys.argv[1]
ckpt = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, code)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixsched  # noqa: E402
fixsched.apply()
from ali2ali import Ali2Ali  # noqa: E402

m = Ali2Ali(lora_rank_unet=8, lora_rank_vae=4)
if ckpt:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    for mod, key in ((m.unet, "state_dict_unet"), (m.vae, "state_dict_vae")):
        cur = mod.state_dict(); cur.update(sd[key]); mod.load_state_dict(cur)
m.set_eval()
tok = lambda p: m.tokenizer(p, max_length=m.tokenizer.model_max_length, padding="max_length",
                            truncation=True, return_tensors="pt").input_ids.cuda()
t1, t2 = tok("Add a crack."), tok("Add oil.")
print("token ids differ:", bool((t1 != t2).any()))
with torch.no_grad():
    e1, e2 = m.text_encoder(t1)[0], m.text_encoder(t2)[0]
print("text embeddings |diff| mean:", (e1 - e2).abs().mean().item())
x = torch.randn(1, 4, 64, 64, device="cuda")
with torch.no_grad():
    u1 = m.unet(x, m.timesteps, encoder_hidden_states=e1).sample
    u2 = m.unet(x, m.timesteps, encoder_hidden_states=e2).sample
print("U-Net output |diff| mean:", (u1 - u2).abs().mean().item(), "| |u1| mean:", u1.abs().mean().item())
# which attention processors does the U-Net use for cross-attention?
procs = {type(p).__name__ for n, p in m.unet.attn_processors.items() if "attn2" in n}
print("attn2 processors:", procs)
print("unet.config.cross_attention_dim:", m.unet.config.cross_attention_dim, "| e1 shape:", tuple(e1.shape))
