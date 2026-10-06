#!/bin/bash
# run_judge_train.sh — label fidelity of the synthetic sets actually used for CAS/NAS (training masks),
# review 2026-10-05. Same judge as the held-out study (EfficientNet-B0 on real training crops, 3 seeds),
# applied to the training pools of both carriers. Local RTX 5090, env D:/aliaug_env.
PY=D:/aliaug_env/Scripts/python.exe
R=C:/Users/aliha/Desktop/PHD/FIRSTPAPER-FINAL
LIVE=C:/Users/aliha/Desktop/PHD/t23_runs/live
mkdir -p $LIVE/results_judge_train
for C in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw transistor wood zipper; do
  for V in cattext catcolor; do
    for s in 0 1 2; do
      OUT=$LIVE/results_judge_train/${V}_train_${C}_seed${s}.json
      [ -f $OUT ] && continue
      $PY $R/prep/t23/judge.py --split D:/ad/aliaug_cat/$C/split_0 --syn D:/sam3/trainpools/${V}_syn_$C \
        --fold train --seed $s --out $OUT > $LIVE/results_judge_train/${V}_train_${C}_seed${s}.log 2>&1 || echo "FAIL $V $C $s"
    done
  done
  echo "judge-train $C done $(date +%H:%M)"
done
echo ALLDONE
