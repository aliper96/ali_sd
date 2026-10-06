#!/bin/bash
# run_local_nas.sh — NAS detectors for the control pools of make_pools.py (review 2026-10-05):
#   comp_cattext / comp_catcolor (post-hoc compositing) and paste (copy-paste baseline).
# Same detector settings as the per-category study: YOLOv8n-seg, 3 seeds, last epoch; epochs/imgsz
# come from eval/epochs_override.json (rule "/results_cat" -> 50 epochs, 512 px; the output dirs below
# contain that substring on purpose). Local RTX 5090, env D:/aliaug_env.
PY=D:/aliaug_env/Scripts/python.exe
R=C:/Users/aliha/Desktop/PHD/FIRSTPAPER-FINAL
LIVE=C:/Users/aliha/Desktop/PHD/t23_runs/live
POOLS=D:/pools
export YOLO_WORKERS=2   # 8 workers exhausted the Windows page file (WinError 1455) and hung the chain
WORK=D:/pools/_work
for C in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper; do
  SPLIT=D:/ad/aliaug_cat/$C/split_0
  for P in comp_cattext comp_catcolor paste; do
    POOL=$POOLS/${P}_$C
    [ -f $POOL/prompts.json ] || { echo "no pool $POOL"; continue; }
    OUTD=$LIVE/results_cat${P}/$C/multiclass
    [ -f $OUTD/NAS.csv ] && continue
    mkdir -p $OUTD
    Y=$WORK/yolo_${P}_$C
    $PY $R/prep/t23/build_yolo.py --split $SPLIT --poly-split $SPLIT --label-from prompt --syn $POOL --out $Y > $OUTD/build.log 2>&1 || { echo "FAIL build $P $C"; continue; }
    D=$Y/multiclass
    NAMES=$(cat $Y/names_multiclass.txt | tr '\n' ' ')
    $PY $R/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
      --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
      --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
      --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
      --names $NAMES --protocol NAS --ratio 1 --seeds 0 1 2 --epochs 100 --imgsz 640 \
      --workdir $WORK/run_${P}_$C --output-csv $OUTD/NAS.csv > $OUTD/nas.log 2>&1 || echo "FAIL nas $P $C"
    rm -rf $WORK/run_${P}_$C
    echo "$P $C done $(date +%H:%M)"
  done
done
echo ALLDONE
