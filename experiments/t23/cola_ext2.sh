#!/bin/bash
# cola_ext2.sh — full-MVTec text-vs-colour (alltext/allcolor, K=3) + tile component ablation (K=3).
# Starts after the tile textfix/colorfix queue (logs/fix_all.done). 3 chains at a time, interleaved so
# every split's comparison fills in early. Bulky data lives on the local NVMe (/mnt/scratch), which is
# WIPED when the VM stops: results go to the boot disk (results_*, results_judge) and must be
# downloaded before stopping.
R=/home/aliha/t23
S=/mnt/scratch/t23
cd $R
while [ ! -f logs/fix_all.done ]; do sleep 300; done
# splits: aliaug_splits_all is uploaded to $S by the caller; derive colour and no-label versions
[ -d $S/aliaug_splits_all_color ] || python3 FIRSTPAPER-FINAL/prep/t23/build_color_splits.py \
  --src $S/aliaug_splits_all --dst $S/aliaug_splits_all_color
[ -d $S/aliaug_splits_nolabel ] || python3 FIRSTPAPER-FINAL/prep/t23/build_color_splits.py \
  --src $R/aliaug_splits --dst $S/aliaug_splits_nolabel --white
for k in 0 1 2; do
  for V in alltext allcolor abl_noskip abl_noclip abl_nogan abl_nolabel; do echo "$V $k"; done
done | xargs -P 3 -L 1 bash -c 'bash FIRSTPAPER-FINAL/prep/t23/run_variant.sh $0 $1 > logs/${0}_run_$1.log 2>&1'
echo "EXT2 DONE" > logs/ext2_all.done
