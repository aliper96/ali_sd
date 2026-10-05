#!/bin/bash
# run_cat.sh <carrier> <category> — per-category LoRA study (split 0 of the full-MVTec splits).
#   carrier = cattext (class by prompt) | catcolor (class by mask colour, constant prompt)
# Same recipe as tile (textfix/colorfix): released code + one-step scheduler, 10k steps, no dropout,
# LoRA 8/4, seed 0, final checkpoint. Downstream: multiclass task within the category (as the tile
# multiclass table), YOLOv8n-seg, 3 detector seeds, last.pt on the held-out real test fold; detector
# epochs and image size come from eval/epochs_override.json (rule "/results_cat"). The real-only D_S_AUG
# reference is run once per category, in the cattext chain. Then the E2 judge: held-out + swap.
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
G=$S/${V}_gen_$C
CK=$G/checkpoints/model_10000.pkl
[ -f $CK ] || python3 -u FIRSTPAPER-FINAL/prep/t23/train_gen.py --code $R/ali_sd_public --fix-sched \
  --split $SPLIT --out $G --seed 0 --workers 2 --steps 10000 > $R/logs/${V}_${C}_train_gen.log 2>&1
[ -f $S/${V}_syn_$C/prompts.json ] || python3 -u FIRSTPAPER-FINAL/prep/t23/gen_syn.py --code $R/ali_sd_public \
  --fix-sched --split $SPLIT --ckpt $CK --out $S/${V}_syn_$C
python3 -u FIRSTPAPER-FINAL/prep/t23/build_yolo.py --split $SPLIT --poly-split $POLY --label-from $LABEL \
  --syn $S/${V}_syn_$C --out $S/${V}_yolo_$C
D=$S/${V}_yolo_$C/multiclass
NAMES=$(cat $S/${V}_yolo_$C/names_multiclass.txt | tr '\n' ' ')
PROTOS="CAS NAS"; [ $V = cattext ] && PROTOS="D_S_AUG CAS NAS"
for P in $PROTOS; do
  OUTD=$R/results_$V/$C/multiclass; [ $P = D_S_AUG ] && OUTD=$R/results_catreal/$C/multiclass
  [ -f $OUTD/$P.csv ] && continue
  python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
    --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
    --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
    --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
    --names $NAMES --protocol $P --ratio 1 --seeds 0 1 2 --epochs 100 --imgsz 640 \
    --workdir $S/work/${V}_$C/$P --output-csv $OUTD/$P.csv
  rm -rf $OUTD/_yolo_runs/*/weights $S/work/${V}_$C/$P
done
for mode in heldout swap; do
  OUT=$S/judge_syn/${V}_${mode}_$C
  SW=""; [ $mode = swap ] && SW="--swap"
  [ -f $OUT/labels.json ] || python3 -u FIRSTPAPER-FINAL/prep/t23/gen_syn.py --code $R/ali_sd_public --fix-sched \
    --split $SPLIT --ckpt $CK --out $OUT --fold test $SW
  for s in 0 1 2; do
    python3 FIRSTPAPER-FINAL/prep/t23/judge.py --split $POLY --syn $OUT --fold test --seed $s \
      --out $R/results_judge/${V}_${mode}_${C}_seed${s}.json
  done
done
echo "$V $C DONE"
