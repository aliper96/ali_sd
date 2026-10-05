#!/bin/bash
# run_split.sh <k> — everything after generator training for split k (A100).
#   1. wait for gen_split_k/checkpoints/model_10000.pkl (final checkpoint, fixed a priori)
#   2. generate the synthetic pool from the TRAIN fold (gen_syn.py)
#   3. build binary + multiclass YOLO-seg datasets (build_yolo.py)
#   4. D_S, D_S_AUG, CAS, NAS x {binary, multiclass} x seeds {0,1,2}, evaluated on the
#      held-out real test fold with last.pt (run_downstream_yolo.py)
# YOLO settings fixed for every cell: yolov8n-seg, 100 epochs, imgsz 640.
set -e
k=$1
R=/home/aliha/t23
cd $R
export PYTHONPATH=/home/aliha/aliaug/libs
CK=$R/gen_split_$k/checkpoints/model_10000.pkl
while [ ! -f $CK ]; do sleep 120; done
sleep 30

python3 -u FIRSTPAPER-FINAL/prep/t23/gen_syn.py --code $R/ali_sd_public \
  --split $R/aliaug_splits/split_$k --ckpt $CK --out $R/syn_split_$k
python3 -u FIRSTPAPER-FINAL/prep/t23/build_yolo.py --split $R/aliaug_splits/split_$k \
  --syn $R/syn_split_$k --out $R/yolo_split_$k

for task in binary multiclass; do
  D=$R/yolo_split_$k/$task
  NAMES=$(cat $R/yolo_split_$k/names_$task.txt | tr '\n' ' ')
  for P in D_S D_S_AUG CAS NAS; do
    python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
      --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
      --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
      --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
      --names $NAMES --protocol $P --ratio 1 --seeds 0 1 2 --epochs 100 --imgsz 640 \
      --workdir $R/work/split_$k/$task \
      --output-csv $R/results/split_$k/$task/$P.csv
  done
done
echo "split $k DONE"
