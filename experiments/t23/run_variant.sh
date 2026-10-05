#!/bin/bash
# run_variant.sh <variant> <k> — full chain for one split of a generator variant (A100).
#   textfix   : tile, released code + set_timesteps(1) restored (fixsched.py), class by TEXT prompt
#   colorfix  : tile, same, class by MASK COLOUR with a constant prompt
#   color     : tile, colour with the released (broken-scheduler) code — kept for the record
#   alltext   : ALL MVTec-AD (15 categories, 73 types, one adapter), text, 20k steps
#   allcolor  : ALL MVTec-AD, colour, 20k steps
#   abl_noskip / abl_noclip / abl_nogan : tile colorfix with one component removed (theme I)
#   abl_nolabel : tile, white mask + constant prompt: NO label carrier (class only from mask shape)
# Protocol as run_split.sh: generator (no dropout, seed = k) -> synthetics from the train fold ->
# CAS/NAS, binary + multiclass YOLO-seg, 3 seeds, last.pt on the held-out test fold (D_S / D_S_AUG are
# real-only and identical for every variant of the same data: not re-run here) -> E2 judge on held-out
# test-fold masks (own class, and swap when the variant has a label carrier).
set -e
V=$1; k=$2
R=/home/aliha/t23          # code, logs, results (boot disk)
S=/mnt/scratch/t23         # bulky data of the new variants (local NVMe, wiped when the VM stops)
cd $R
export PYTHONPATH=/home/aliha/aliaug/libs
FIX="--fix-sched"; EXTRA_TRAIN=""; EXTRA_GEN=""; STEPS=10000; LABEL=name; SWAP=1; DATA=$S
POLY=$R/aliaug_splits
case $V in
  textfix)     SPLITS=$R/aliaug_splits;       LABEL=prompt; DATA=$R ;;
  colorfix)    SPLITS=$R/aliaug_splits_color;               DATA=$R ;;
  color)       SPLITS=$R/aliaug_splits_color; FIX="";       DATA=$R ;;
  alltext)     SPLITS=$S/aliaug_splits_all;       LABEL=prompt; STEPS=20000; POLY=$S/aliaug_splits_all ;;
  allcolor)    SPLITS=$S/aliaug_splits_all_color;               STEPS=20000; POLY=$S/aliaug_splits_all ;;
  abl_noskip)  SPLITS=$R/aliaug_splits_color; EXTRA_TRAIN="--no-skip"; EXTRA_GEN="--no-skip" ;;
  abl_noclip)  SPLITS=$R/aliaug_splits_color; EXTRA_TRAIN="--lambda-clip 0" ;;
  abl_nogan)   SPLITS=$R/aliaug_splits_color; EXTRA_TRAIN="--lambda-gan 0" ;;
  abl_nolabel) SPLITS=$S/aliaug_splits_nolabel; SWAP=0 ;;
  *) echo "unknown variant $V"; exit 1 ;;
esac
mkdir -p $DATA
G=$DATA/${V}_gen_split_$k
CK=$G/checkpoints/model_$STEPS.pkl
[ -f $CK ] || python3 -u FIRSTPAPER-FINAL/prep/t23/train_gen.py --code $R/ali_sd_public $FIX $EXTRA_TRAIN \
  --split $SPLITS/split_$k --out $G --seed $k --workers 2 --steps $STEPS > $R/logs/${V}_train_gen_$k.log 2>&1
[ -f $DATA/${V}_syn_split_$k/prompts.json ] || python3 -u FIRSTPAPER-FINAL/prep/t23/gen_syn.py \
  --code $R/ali_sd_public $FIX $EXTRA_GEN --split $SPLITS/split_$k --ckpt $CK --out $DATA/${V}_syn_split_$k
python3 -u FIRSTPAPER-FINAL/prep/t23/build_yolo.py --split $SPLITS/split_$k \
  --poly-split $POLY/split_$k --label-from $LABEL \
  --syn $DATA/${V}_syn_split_$k --out $DATA/${V}_yolo_split_$k
for task in binary multiclass; do
  D=$DATA/${V}_yolo_split_$k/$task
  NAMES=$(cat $DATA/${V}_yolo_split_$k/names_$task.txt | tr '\n' ' ')
  for P in CAS NAS; do
    [ -f $R/results_${V}/split_$k/$task/$P.csv ] && continue
    python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
      --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
      --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
      --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
      --names $NAMES --protocol $P --ratio 1 --seeds 0 1 2 --epochs 100 --imgsz 640 \
      --workdir $DATA/work/${V}_split_$k/$task \
      --output-csv $R/results_${V}/split_$k/$task/$P.csv
    rm -rf $R/results_${V}/split_$k/$task/_yolo_runs/*/weights $DATA/work/${V}_split_$k/$task
  done
done
# E2: label fidelity on held-out masks, own class and (if there is a label carrier) swapped class
MODES="heldout"; [ $SWAP = 1 ] && MODES="heldout swap"
for mode in $MODES; do
  OUT=$DATA/judge_syn/${V}_${mode}_split_$k
  SW=""; [ $mode = swap ] && SW="--swap"
  [ -f $OUT/labels.json ] || python3 -u FIRSTPAPER-FINAL/prep/t23/gen_syn.py --code $R/ali_sd_public $FIX $EXTRA_GEN \
    --split $SPLITS/split_$k --ckpt $CK --out $OUT --fold test $SW
  for s in 0 1 2; do
    python3 FIRSTPAPER-FINAL/prep/t23/judge.py --split $POLY/split_$k --syn $OUT \
      --fold test --seed $s --out $R/results_judge/${V}_${mode}_split${k}_seed${s}.json
  done
done
echo "$V split $k DONE"
