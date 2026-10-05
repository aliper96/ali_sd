#!/bin/bash
# run_clean.sh — secondary analysis fixed 2026-10-03 13:00, before any result on this subset: re-train and
# evaluate the four detectors on the split-0 test images that AnomalyDiffusion's released generator NEVER saw
# (index > n_type // 3, the complement of its training third). Conditions: real-only (D_S_AUG), Ali-AUG text
# NAS, Ali-AUG colour NAS, AD NAS; same YOLO settings (rule "/results_clean/": 50 ep, 512 px), 3 seeds.
# Runs each category once its AD chain is done (logs/ad_cat.log "AD <cat> DONE"). End: logs/clean_all.done
R=/home/aliha/t23
S=/mnt/scratch/t23
T23=$R/FIRSTPAPER-FINAL/prep/t23
cd $R
export PYTHONPATH=/home/aliha/aliaug/libs
CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper"
for C in $CATS; do
  until grep -q "AD $C DONE" logs/ad_cat.log; do sleep 300; done
  for V in real cattext catcolor ad; do
    case $V in
      real)     Y=$S/cattext_yolo_$C;  P=D_S_AUG ;;
      cattext)  Y=$S/cattext_yolo_$C;  P=NAS ;;
      catcolor) Y=$S/catcolor_yolo_$C; P=NAS ;;
      ad)       Y=$S/ad_yolo_$C;       P=NAS ;;
    esac
    D=$Y/multiclass
    CV=$S/clean_val/${V}_$C
    if [ ! -d $CV/images ]; then
      mkdir -p $CV/images $CV/labels
      python3 - "$D/real_test" "$CV" <<'PY'
import os, sys, collections
sys.path.insert(0, "/home/aliha/t23/FIRSTPAPER-FINAL/prep/t23")
from mvtec_names import parse
src, dst = sys.argv[1], sys.argv[2]
cat = parse(sorted(os.listdir(f"{src}/images"))[0])[0]
sp = f"/mnt/scratch/t23/aliaug_cat/{cat}/split_0"
n_type = collections.Counter(parse(n)[1] for f in ("train_A", "test_A") for n in os.listdir(f"{sp}/{f}"))
k = 0
for n in sorted(os.listdir(f"{src}/images")):
    if int(os.path.splitext(n)[0].rsplit("_", 1)[1]) > n_type[parse(n)[1]] // 3:
        os.symlink(os.path.realpath(f"{src}/images/{n}"), f"{dst}/images/{n}")
        lb = os.path.splitext(n)[0] + ".txt"
        os.symlink(os.path.realpath(f"{src}/labels/{lb}"), f"{dst}/labels/{lb}")
        k += 1
print(cat, "unseen test images:", k)
PY
    fi
    NAMES=$(cat $Y/names_multiclass.txt | tr '\n' ' ')
    OUT=$R/results_clean/$V/$C/multiclass/$P.csv
    [ -f $OUT ] && continue
    python3 -u FIRSTPAPER-FINAL/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
      --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
      --val-images-dir $CV/images --val-labels-dir $CV/labels \
      --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
      --names $NAMES --protocol $P --ratio 1 --seeds 0 1 2 --epochs 100 --imgsz 640 \
      --workdir $S/work/clean_${V}_$C --output-csv $OUT
    rm -rf $(dirname $OUT)/_yolo_runs/*/weights $S/work/clean_${V}_$C
  done
  echo "CLEAN $C DONE"
done
echo "CLEAN DONE" > logs/clean_all.done
