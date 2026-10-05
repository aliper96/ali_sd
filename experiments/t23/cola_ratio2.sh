#!/bin/bash
# cola_ratio2.sh — replaces cola_ratio.sh after the 2026-10-03 16:00 OOM cascade (4 chains failed in 60 s because
# the GPU was full and a failure was marked as done). Same chains and run_ratio.sh; now a chain starts only with
# >= 14 GB GPU memory free, a failed chain is retried (up to 3 attempts, 20 min apart) and only a success writes
# logs/ratio_<V>_<cat>.done. End: logs/ratio_all.done
R=/home/aliha/t23
cd $R
CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for c in $CATS; do for v in cattext catcolor; do
  [ -f logs/ratio_${v}_${c}.done ] && continue
  until grep -q DONE logs/${v}_${c}_run.log 2>/dev/null; do sleep 600; done
  for attempt in 1 2 3; do
    until [ "$(free_mb)" -ge 14000 ]; do sleep 120; done
    echo "$(date '+%F %T') start $v $c (attempt $attempt)" >> logs/cola_ratio.log
    if bash FIRSTPAPER-FINAL/prep/t23/run_ratio.sh $v $c > logs/ratio_${v}_${c}.log 2>&1 < /dev/null; then
      touch logs/ratio_${v}_${c}.done; echo "$(date '+%F %T') done $v $c" >> logs/cola_ratio.log; break
    fi
    echo "$(date '+%F %T') FAILED $v $c (attempt $attempt)" >> logs/cola_ratio.log
    rm -f /mnt/scratch/t23/${v}_syn5_$c/labels.json
    sleep 1200
  done
done; done
echo "RATIO DONE" > logs/ratio_all.done
