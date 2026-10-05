"""
ablations.py — component ablations of Ali-AUG (review theme I), applied to the released model code.

no_skip : disable the skip connections from the VAE encoder to the decoder (the path that carries the
          mask — and the input image details — into the decoder in the released code). Implemented by
          setting vae.decoder.ignore_skip = True after the model is built, which makes the decoder run
          without the skip branch (utils.my_vae_decoder_fwd).
The loss ablations (no CLIP similarity, no GAN) need no patch: they are the trainer's own
--lambda_clipsim 0 / --lambda_gan 0.

Call apply_no_skip() after the released code is on sys.path; it patches the Ali2Ali class itself, so it
also affects code that imported the class earlier. Training and generation must use the same setting.
"""


def apply_no_skip():
    import ali2ali
    cls = ali2ali.Ali2Ali
    if getattr(cls, "_no_skip", False):
        return
    orig = cls.__init__

    def init(self, *x, **k):
        orig(self, *x, **k)
        self.vae.decoder.ignore_skip = True

    cls.__init__ = init
    cls._no_skip = True
