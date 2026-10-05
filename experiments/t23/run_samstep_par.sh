#!/bin/bash
# run_samstep_par.sh — same work as run_samstep.sh, but safe to run as several parallel workers: each (carrier,
# category) chain is claimed with an atomic mkdir lock (logs/samstep_locks/<V>_<C>), so no chain runs twice.
#   for w in 1 2 3; do setsid nohup bash run_samstep_par.sh > logs/samstep_w$w.out 2>&1 & done
R=/home/aliha/t23
S=/mnt/scratch/t23
cd $R
export PYTHONPATH=/home/aliha/aliaug/libs
mkdir -p logs/samstep_locks
for C in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper; do
  for V in cattext catcolor; do
    OUT=$R/results_samstep_$V/$C/multiclass/NAS.csv
    [ -f $OUT ] && continue
    mkdir logs/samstep_locks/${V}_$C 2>/dev/null || continue
    case $V in
      cattext)  SPLIT=$S/aliaug_cat/$C/split_0;       LABEL=prompt ;;
      catcolor) SPLIT=$S/aliaug_cat_color/$C/split_0; LABEL=name ;;
    esac
    python3 -u FIRSTPAPER-FINAL/prep/t23/build_yolo.py --split $SPLIT --poly-split $S/aliaug_cat/$C/split_0 \
      --label-from $LABEL --syn $S/${V}_syn_$C --syn-masks $S/sam_masks/${V}_$C --drop-missing \
      --out $S/${V}_yolosam_$C > logs/samstep_build_${V}_$C.log 2>&1 || { echo "build failed $V $C" >> logs/samstep.log; continue; }
    D=$S/${V}_yolosam_$C/multiclass
    NAMES=$(cat $S/${V}_yolosam_$C/names_multiclass.txt | tr '\n' ' ')
    python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
      --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
      --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
      --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
      --names $NAMES --protocol NAS --ratio 1 --seeds 0 1 2 --epochs 100 --imgsz 640 \
      --workdir $S/work/samstep_${V}_$C --output-csv $OUT > logs/samstep_${V}_$C.log 2>&1
    rm -rf $(dirname $OUT)/_yolo_runs/*/weights $S/work/samstep_${V}_$C
    echo "$(date '+%F %T') samstep $V $C done" >> logs/samstep.log
  done
done
ls results_samstep_cattext/*/multiclass/NAS.csv results_samstep_catcolor/*/multiclass/NAS.csv 2>/dev/null | wc -l \
  | grep -q '^28$' && echo "SAMSTEP DONE" > logs/samstep_all.done
