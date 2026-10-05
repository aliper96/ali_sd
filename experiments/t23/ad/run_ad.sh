#!/bin/bash
# run_ad.sh — AnomalyDiffusion baseline under our protocol (fixed 2026-10-02 before any AD result).
#  1. data: our split-0 train fold only, all 14 categories of the per-category study (build_ad_data.py)
#  2. train: official main.py and recipe (spatial encoder + anomaly embeddings, LDM frozen, batch 4,
#     lr 5e-3, data_enhance), 300k iterations as in their paper; ONE model for all categories as in their
#     paper; final checkpoint (no selection). Their 1/3 split is replaced by our train fold and the StyleGAN2
#     CUDA ops by their pure-PyTorch formulas (patch_ad.py; nothing else changed).
#  3. generate: one image per train mask with our clean inputs (gen_ad.py), DDIM 200 (their default)
#  4. downstream: identical to Ali-AUG (run_ad_cat.sh: build_yolo + NAS/CAS, 50 ep, 512 px, 3 seeds)
set -e
AD=/mnt/scratch/ad
REPO=$AD/anomalydiffusion
T23=/home/aliha/t23/FIRSTPAPER-FINAL/prep/t23
cd $REPO
. $AD/venv/bin/activate
python $T23/ad/patch_ad.py $REPO
[ -f $AD/mvtec_ours/manifest.json ] || python $T23/ad/build_ad_data.py --cat-root /mnt/scratch/t23/aliaug_cat \
  --out $AD/mvtec_ours --name-list $REPO/name-anomaly.txt
# main.py always writes to <logdir>/anomaly-checkpoints; a dedicated logdir keeps the smoke run apart
CK=logs_ours/anomaly-checkpoints/checkpoints
if ! ls $CK/embeddings_gs-299*.pt $CK/embeddings_gs-300000.pt >/dev/null 2>&1; then
  AD_ALL=1 AD_NATIVE_OPS=1 python main.py --spatial_encoder_embedding --data_enhance \
    --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t \
    --actual_resume models/ldm/text2img-large/model.ckpt -n ours_split0 --logdir logs_ours --gpus 0, \
    --init_word anomaly --mvtec_path=$AD/mvtec_ours --max_steps 300000
fi
ls -la $CK | tail -5
echo "AD TRAIN DONE $REPO/$CK"
