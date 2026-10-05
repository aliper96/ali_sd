#!/bin/bash
# cola_fix.sh — the corrected-scheduler experiments (textfix, colorfix), 3 chains at a time.
# Starts once the last broken-scheduler text generators (splits 3, 4) have finished, so the GPU
# is not over-committed. Jobs interleave text and colour of the same split, so paired
# text-vs-colour comparisons accumulate from the start.
R=/home/aliha/t23
cd $R
for k in 3 4; do
  while [ ! -f gen_split_$k/checkpoints/model_10000.pkl ]; do sleep 120; done
done
for k in 0 1 2 3 4; do echo "textfix $k"; echo "colorfix $k"; done | \
  xargs -P 3 -L 1 bash -c 'bash FIRSTPAPER-FINAL/prep/t23/run_variant.sh $0 $1 > logs/${0}_run_$1.log 2>&1'
echo "FIX DONE" > logs/fix_all.done
