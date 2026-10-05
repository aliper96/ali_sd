#!/bin/bash
# run_ad_cat.sh — downstream of the AnomalyDiffusion baseline, per category, identical to run_cat.sh (cattext):
# synthetic pool = one image per TRAIN mask of split 0 (gen_ad.py, our clean inputs, label = requested type),
# YOLOv8n-seg multiclass within the category, NAS (1:1) and CAS, 3 detector seeds, last.pt on the held-out
# real test fold; detector settings from eval/epochs_override.json (rule "/results_ad/": 50 ep, 512 px, same
# as "/results_cat"). Real-only reference = results_catreal (shared with Ali-AUG). Then the judge on
# held-out masks (gen_ad.py --fold test). Usage: run_ad_cat.sh <ckpt_dir> [categories...]
set -e
CK=$1; shift
CATS=${@:-"bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper"}
R=/home/aliha/t23
S=/mnt/scratch/t23
AD=/mnt/scratch/ad
T23=$R/FIRSTPAPER-FINAL/prep/t23
cd $R
for C in $CATS; do
  SPLIT=$S/aliaug_cat/$C/split_0
  SYN=$S/ad_syn_$C
  [ -f $SYN/prompts.json ] || ( cd $AD/anomalydiffusion && . $AD/venv/bin/activate && \
    AD_NATIVE_OPS=1 python $T23/ad/gen_ad.py --split $SPLIT --ckpt-dir $CK --out $SYN )
  PYTHONPATH=/home/aliha/aliaug/libs python3 -u $T23/build_yolo.py --split $SPLIT --poly-split $SPLIT \
    --label-from prompt --syn $SYN --out $S/ad_yolo_$C
  D=$S/ad_yolo_$C/multiclass
  NAMES=$(cat $S/ad_yolo_$C/names_multiclass.txt | tr '\n' ' ')
  OUTD=$R/results_ad/$C/multiclass
  for P in CAS NAS; do
    [ -f $OUTD/$P.csv ] && continue
    PYTHONPATH=/home/aliha/aliaug/libs python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment \
      --model yolov8n-seg.pt \
      --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
      --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
      --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
      --names $NAMES --protocol $P --ratio 1 --seeds 0 1 2 --epochs 100 --imgsz 640 \
      --workdir $S/work/ad_$C/$P --output-csv $OUTD/$P.csv
    rm -rf $OUTD/_yolo_runs/*/weights $S/work/ad_$C/$P
  done
  if [ $C != toothbrush ]; then  # one defect type: no judge (as for Ali-AUG)
    OUT=$S/judge_syn/ad_heldout_$C
    [ -f $OUT/labels.json ] || ( cd $AD/anomalydiffusion && . $AD/venv/bin/activate && \
      AD_NATIVE_OPS=1 python $T23/ad/gen_ad.py --split $SPLIT --ckpt-dir $CK --out $OUT --fold test )
    for s in 0 1 2; do
      PYTHONPATH=/home/aliha/aliaug/libs python3 $T23/judge.py --split $SPLIT --syn $OUT --fold test --seed $s \
        --out $R/results_judge/ad_heldout_${C}_seed${s}.json
    done
  fi
  echo "AD $C DONE"
done
echo "AD ALL DONE" > logs/ad_all.done
