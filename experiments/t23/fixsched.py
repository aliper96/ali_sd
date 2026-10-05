"""
fixsched.py — restore the one-step scheduler of the released Ali-AUG code.

In ali_sd_public/utils.py the line `noise_scheduler_1step.set_timesteps(1, device="cuda")` is
commented out. Without it, DDPMScheduler.step() at t=999 steps to t=998 instead of to x0, so the
U-Net output enters the result with a coefficient of ~1e-3: the U-Net — the only path of the text
prompt — is effectively disconnected (measured with prep/t23/diag_unet.py on a trained checkpoint:
changing the prompt changes the output by 0.000 grey levels; zeroing the U-Net output by <0.31).
Img2Img-turbo, which Ali-AUG builds on, calls set_timesteps(1). This module re-enables it.

Call apply() AFTER putting the released code on sys.path and BEFORE importing ali2ali /
train_ali2ali (ali2ali does `from utils import make_1step_sched` at import time).
"""


def apply():
    import utils
    orig = utils.make_1step_sched
    if getattr(orig, "_fixed", False):
        return

    def make_1step_sched_fixed():
        s = orig()
        s.set_timesteps(1, device="cuda")
        return s

    make_1step_sched_fixed._fixed = True
    utils.make_1step_sched = make_1step_sched_fixed
