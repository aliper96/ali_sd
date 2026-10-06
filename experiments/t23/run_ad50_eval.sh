#!/bin/bash
# run_ad50_eval.sh — evaluate AnomalyDiffusion at 50 DDIM steps under our protocol (review 2026-10-05):
# judge on its held-out pools (ad50_heldout_<cat>) and NAS detectors on its training pools (ad50_syn/<cat>).
# Waits for gen_ad50.sh to finish. Output dirs contain "/results_cat" so epochs_override gives 50 epochs / 512 px.
PY=D:/aliaug_env/Scripts/python.exe
R=C:/Users/aliha/Desktop/PHD/FIRSTPAPER-FINAL
LIVE=C:/Users/aliha/Desktop/PHD/t23_runs/live
WORK=D:/pools/_work
until grep -q ALLDONE /d/ad/gen_ad50_b.log 2>/dev/null; do sleep 60; done
for C in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw transistor wood zipper; do
  for s in 0 1 2; do
    OUT=$LIVE/results_judge/ad50_heldout_${C}_seed${s}.json
    [ -f $OUT ] && continue
    [ -f D:/ad/judge_syn/ad50_heldout_$C/labels.json ] || { echo "no ad50 heldout pool $C"; break; }
    $PY $R/prep/t23/judge.py --split D:/ad/aliaug_cat/$C/split_0 --syn D:/ad/judge_syn/ad50_heldout_$C --fold test --seed $s --out $OUT > $LIVE/results_judge/ad50_heldout_${C}_seed${s}.log 2>&1 || echo "FAIL judge $C $s"
  done
done
for C in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper; do
  SPLIT=D:/ad/aliaug_cat/$C/split_0; POOL=D:/ad/ad50_syn/$C
  [ -f $POOL/labels.json ] || { echo "no ad50 pool $C"; continue; }
  OUTD=$LIVE/results_cat_ad50/$C/multiclass
  [ -f $OUTD/NAS.csv ] && continue
  mkdir -p $OUTD
  Y=$WORK/yolo_ad50_$C
  $PY $R/prep/t23/build_yolo.py --split $SPLIT --poly-split $SPLIT --label-from prompt --syn $POOL --out $Y > $OUTD/build.log 2>&1 || { echo "FAIL build ad50 $C"; continue; }
  D=$Y/multiclass; NAMES=$(cat $Y/names_multiclass.txt | tr '\n' ' ')
  $PY $R/eval/run_downstream_yolo.py --task segment --model yolov8n-seg.pt \
    --real-images-dir $D/real_train/images --real-labels-dir $D/real_train/labels \
    --val-images-dir $D/real_test/images --val-labels-dir $D/real_test/labels \
    --syn-images-dir $D/syn/images --syn-labels-dir $D/syn/labels \
    --names $NAMES --protocol NAS --ratio 1 --seeds 0 1 2 --epochs 100 --imgsz 640 \
    --workdir $WORK/run_ad50_$C --output-csv $OUTD/NAS.csv > $OUTD/nas.log 2>&1 || echo "FAIL nas ad50 $C"
  rm -rf $WORK/run_ad50_$C
  echo "ad50 $C done $(date +%H:%M)"
done
echo ALLDONE
