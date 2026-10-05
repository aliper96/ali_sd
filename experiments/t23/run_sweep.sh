#!/bin/bash
# run_sweep.sh <k> — E3: synthetic-to-real ratio sweep (review theme I) for split k (A100).
# Uses the TEXT generator already trained for Tables 2-3 (no new training). 5 synthetic images
# per training mask (300) -> NAS at ratio 0.5, 1, 2, 5 x seeds 0,1,2, binary + multiclass,
# evaluated like Tables 2-3 (held-out real test fold, last.pt). The D_S_AUG reference is the
# one already computed for Tables 2-3 (identical real data and seeds).
set -e
k=$1
R=/home/aliha/t23
cd $R
export PYTHONPATH=/home/aliha/aliaug/libs
while ! grep -q "split $k DONE" logs/run_split_$k.log 2>/dev/null; do sleep 120; done
python3 -u FIRSTPAPER-FINAL/prep/t23/gen_syn.py --code $R/ali_sd_public \
  --split $R/aliaug_splits/split_$k --ckpt $R/gen_split_$k/checkpoints/model_10000.pkl \
  --out $R/sweep_syn_split_$k --per-mask 5
python3 -u FIRSTPAPER-FINAL/prep/t23/build_yolo.py --split $R/aliaug_splits/split_$k \
  --syn $R/sweep_syn_split_$k --out $R/sweep_yolo_split_$k
for task in binary multiclass; do
  D=$R/sweep_yolo_split_$k/$task
  NAMES=$(cat $R/sweep_yolo_split_$k/names_$task.txt | tr '\n' ' ')
  python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
    --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
    --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
    --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
    --names $NAMES --protocol NAS --ratio 0.5 1 2 5 --seeds 0 1 2 --epochs 100 --imgsz 640 \
    --workdir $R/work/sweep_split_$k/$task \
    --output-csv $R/results_sweep/split_$k/$task/NAS_sweep.csv
  rm -rf $R/results_sweep/split_$k/$task/_yolo_runs/*/weights
done
echo "sweep split $k DONE"
