#!/bin/bash
# cola_ext.sh — queue for the extension experiments on the A100.
# E1 (colour variant): starts when the text generators of splits 3 and 4 have finished
# (the GPU then only has YOLO load); splits 0-2 in parallel, then 3-4.
R=/home/aliha/t23
cd $R
for k in 3 4; do
  while [ ! -f gen_split_$k/checkpoints/model_10000.pkl ]; do sleep 120; done
done
[ -d aliaug_splits_color ] || python3 FIRSTPAPER-FINAL/prep/t23/build_color_splits.py \
  --src $R/aliaug_splits --dst $R/aliaug_splits_color
for k in 0 1 2; do bash FIRSTPAPER-FINAL/prep/t23/run_variant.sh color $k > logs/color_run_$k.log 2>&1 & done
wait
for k in 3 4; do bash FIRSTPAPER-FINAL/prep/t23/run_variant.sh color $k > logs/color_run_$k.log 2>&1 & done
wait
echo "E1 DONE"
