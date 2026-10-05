#!/bin/bash
# cola_ad.sh — starts the AnomalyDiffusion baseline once every per-category Ali-AUG job has been started by
# cola_cat.sh (they keep priority) and the GPU has >= 30 GB free. First a 30-step smoke run (separate log
# name, discarded), then the full run_ad.sh. Log: logs/cola_ad.log
R=/home/aliha/t23
AD=/mnt/scratch/ad
cd $R
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
until grep -q "started catcolor zipper" logs/cola_cat.log && [ "$(free_mb)" -ge 30000 ]; do sleep 300; done
echo "$(date '+%F %T') smoke" >> logs/cola_ad.log
( cd $AD/anomalydiffusion && . $AD/venv/bin/activate && AD_ALL=1 AD_NATIVE_OPS=1 python main.py --spatial_encoder_embedding \
    --data_enhance --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t \
    --actual_resume models/ldm/text2img-large/model.ckpt -n smoke_descartado --logdir logs_smoke_descartado --gpus 0, --init_word anomaly \
    --mvtec_path=$AD/mvtec_ours --max_steps 30 ) > logs/ad_smoke.log 2>&1 \
  && echo "$(date '+%F %T') smoke OK" >> logs/cola_ad.log \
  || { echo "$(date '+%F %T') smoke FAILED (see logs/ad_smoke.log)" >> logs/cola_ad.log; exit 1; }
echo "$(date '+%F %T') train start" >> logs/cola_ad.log
bash FIRSTPAPER-FINAL/prep/t23/ad/run_ad.sh > logs/ad_train.log 2>&1 \
  && echo "$(date '+%F %T') train DONE" >> logs/cola_ad.log \
  || echo "$(date '+%F %T') train FAILED" >> logs/cola_ad.log
