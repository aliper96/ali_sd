#!/bin/bash
# run_judge.sh — E2: label fidelity (theme B) for text and colour variants, all splits (A100).
# For each (variant, split), once the generator exists:
#   heldout : generate from the TEST-fold masks (never seen by the generator), own class requested
#   swap    : same masks, the NEXT class requested (prompt or colour) -> does the label channel win?
# then the independent judge (judge.py, trained on real TRAIN crops only), 3 judge seeds each.
# These synthetic images are for measurement only; they never enter any training set.
R=/home/aliha/t23
cd $R
export PYTHONPATH=/home/aliha/aliaug/libs
for V in text color; do
  for k in 0 1 2 3 4; do
    if [ $V = text ]; then CK=$R/gen_split_$k/checkpoints/model_10000.pkl; SP=$R/aliaug_splits/split_$k
    else CK=$R/color_gen_split_$k/checkpoints/model_10000.pkl; SP=$R/aliaug_splits_color/split_$k; fi
    while [ ! -f $CK ]; do sleep 300; done
    for mode in heldout swap; do
      OUT=$R/judge_syn/${V}_${mode}_split_$k
      EXTRA=""; [ $mode = swap ] && EXTRA="--swap"
      [ -f $OUT/labels.json ] || python3 -u FIRSTPAPER-FINAL/prep/t23/gen_syn.py --code $R/ali_sd_public \
        --split $SP --ckpt $CK --out $OUT --fold test $EXTRA
      for s in 0 1 2; do
        python3 FIRSTPAPER-FINAL/prep/t23/judge.py --split $R/aliaug_splits/split_$k --syn $OUT \
          --fold test --seed $s --out $R/results_judge/${V}_${mode}_split${k}_seed${s}.json
      done
    done
  done
done
echo "E2 DONE" > logs/judge_all.done
