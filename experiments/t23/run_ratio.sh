#!/bin/bash
# run_ratio.sh <carrier> <category> — synthetic:real ratio sweep (review theme I) for the per-category LoRAs.
# Uses the FINAL per-category checkpoint of run_cat.sh (no retraining). Pool: 5 images per training mask
# (gen_syn --per-mask 5: textures get 4 extra clean images of the same train fold; objects reuse their own
# inpainted clean image with a different noise seed). NAS at ratios 0.5 / 2 / 5 (ratio 1 = run_cat.sh result,
# pool of 1 per mask), 3 detector seeds, same detector settings as run_cat.sh (rule "/results_cat").
# Protocol fixed 2026-10-02 before any ratio result existed; every ratio is reported.
set -e
V=$1; C=$2
R=/home/aliha/t23
S=/mnt/scratch/t23
cd $R
export PYTHONPATH=/home/aliha/aliaug/libs
case $V in
  cattext)  SPLIT=$S/aliaug_cat/$C/split_0;       LABEL=prompt ;;
  catcolor) SPLIT=$S/aliaug_cat_color/$C/split_0; LABEL=name ;;
  *) echo "unknown carrier $V"; exit 1 ;;
esac
POLY=$S/aliaug_cat/$C/split_0
CK=$S/${V}_gen_$C/checkpoints/model_10000.pkl
P5=$S/${V}_syn5_$C
[ -f $P5/labels.json ] || python3 -u FIRSTPAPER-FINAL/prep/t23/gen_syn.py --code $R/ali_sd_public \
  --fix-sched --split $SPLIT --ckpt $CK --out $P5 --per-mask 5
python3 -u FIRSTPAPER-FINAL/prep/t23/build_yolo.py --split $SPLIT --poly-split $POLY --label-from $LABEL \
  --syn $P5 --out $S/${V}_yolo5_$C
D=$S/${V}_yolo5_$C/multiclass
NAMES=$(cat $S/${V}_yolo5_$C/names_multiclass.txt | tr '\n' ' ')
OUTD=$R/results_catratio_$V/$C/multiclass
[ -f $OUTD/NAS.csv ] || python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
  --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
  --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
  --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
  --names $NAMES --protocol NAS --ratio 0.5 2 5 --seeds 0 1 2 --epochs 100 --imgsz 640 \
  --workdir $S/work/ratio_${V}_$C --output-csv $OUTD/NAS.csv
rm -rf $OUTD/_yolo_runs/*/weights $S/work/ratio_${V}_$C
echo "$V $C RATIO DONE"
