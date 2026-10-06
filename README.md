# Ali-AUG — one-step defect insertion for industrial data augmentation

Code of the paper *Ali-AUG: One-Step Defect Insertion for Industrial Data Augmentation with
Verifiable, Colour-Coded Labels* (Hamza, Lojo, Núñez-Marcos, Atutxa). Ali-AUG inserts a requested defect
into an existing image in a single denoising step, at the location given by a mask, on a largely frozen
one-step Stable-Diffusion backbone (Img2Img-turbo, Parmar et al. 2024) with LoRA adapters, zero-initialised
skip connections and a trainable input convolution. The class of the defect is carried either by a text
prompt or by the colour of the mask.

* `train_ali2ali.py`, `ali2ali.py`, `utils.py`, `inference_ali2ali.py` — the generator (training and inference).
* `experiments/` — everything used for the MVTec-AD studies of the paper: experiment plan, split manifests,
  launch scripts, evaluation code, result files and the generated tables. See `experiments/README.md`.
* `src/` — the Img2Img-turbo code the model builds on.

## Install

```
pip install -r requirements.txt
```

## Data format

A dataset folder contains `train_A` (masks), `train_B` (targets: real defective images), `train_C`
(conditioning inputs: defect-free images, or the target with the masked region removed for pose-varying
objects) and `train_prompts.json` (`{file name: prompt}`), and the same with `test_`. For the colour
carrier the masks in `*_A` are painted with the colour of the defect type and every prompt is
`"Add a defect."` (`experiments/t23/build_color_splits.py`, palette in `experiments/t23/mvtec_names.py`).
Builders for MVTec-AD: `experiments/t23/build_aliaug_splits.py` (tile, 5 splits) and
`experiments/t23/build_mvtec_all_splits.py` + `build_cat_splits.py` (one split per category).

## Train and generate

```
python train_ali2ali.py --dataset_folder <split> --output_dir <out> --max_train_steps 10000 \
    --lora_rank_unet 8 --lora_rank_vae 4 --learning_rate 5e-4 --train_batch_size 1 \
    --lambda_lpips 10 --lambda_l2 10 --lambda_gan 2.5 --lambda_clipsim 5 --seed 0
python experiments/t23/gen_syn.py --help      # one synthetic image per training triplet
```

`experiments/t23/train_gen.py` is the wrapper that ran every training of the paper (fixed seeds, no
tracking). Conditioning dropout (`--input_dropout`, `--mask_dropout`) exists behind a flag and is off by
default; the generators of the paper were trained without it.

## Changes with respect to the first public version

* `utils.py`: the call `noise_scheduler_1step.set_timesteps(1)` was commented out. Without it the
  scheduler step at t=999 goes to t=998 instead of predicting x0, so the U-Net output entered the result
  with a coefficient of about 1e-3 and the prompt had no effect. It is re-enabled; all experiments of
  the paper use the corrected scheduler (`experiments/t23/fixsched.py` applies the same fix at run time).
* `train_ali2ali.py`: training now stops at `--max_train_steps` and saves the final checkpoint (the
  original loop ignored the flag and ran until killed).
* `utils.py`: images are converted to RGB with `Image.convert`, which handles both grayscale and RGB
  inputs (the previous manual conversion failed on RGB files).
* The hard-coded Weights & Biases key has been removed from the trainers (that key is revoked).
  Tracking is opt-in: set `WANDB_API_KEY`; otherwise W&B runs in disabled mode.

## Licence and data

The code is released under the MIT License (see `LICENSE`); the Img2Img-turbo code in `src/` keeps its
original MIT licence. MVTec-AD is distributed by MVTec under CC BY-NC-SA 4.0 and is not redistributed here. The
photovoltaic-panel dataset of the paper is industrial data and cannot be shared.
