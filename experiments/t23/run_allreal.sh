#!/bin/bash
# run_allreal.sh — real-only references (D_S, D_S_AUG) for the full-MVTec splits, so alltext/allcolor
# CAS/NAS can be compared with conventional augmentation on the SAME data, seeds and detector.
# No generator involved: build_yolo is given an empty synthetic set. Waits for fix_all.done.
R=/home/aliha/t23
S=/mnt/scratch/t23
cd $R
while [ ! -f logs/fix_all.done ]; do sleep 300; done
mkdir -p $S/empty_syn && echo "{}" > $S/empty_syn/prompts.json
for k in 0 1 2; do
  Y=$S/allreal_yolo_split_$k
  [ -d $Y ] || python3 FIRSTPAPER-FINAL/prep/t23/build_yolo.py --split $S/aliaug_splits_all/split_$k \
    --syn $S/empty_syn --out $Y > logs/allreal_build_$k.log 2>&1
  for task in binary multiclass; do
    D=$Y/$task
    NAMES=$(cat $Y/names_$task.txt | tr '\n' ' ')
    for P in D_S D_S_AUG; do
      [ -f $R/results_allreal/split_$k/$task/$P.csv ] && continue
      python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
        --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
        --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
        --names $NAMES --protocol $P --seeds 0 1 2 --epochs 100 --imgsz 640 \
        --workdir $S/work/allreal_split_$k/$task \
        --output-csv $R/results_allreal/split_$k/$task/$P.csv >> logs/allreal_$k.log 2>&1
      rm -rf $R/results_allreal/split_$k/$task/_yolo_runs/*/weights $S/work/allreal_split_$k/$task
    done
  done
done
echo "ALLREAL DONE" > logs/allreal.done
