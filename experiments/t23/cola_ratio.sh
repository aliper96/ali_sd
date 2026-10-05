#!/bin/bash
# cola_ratio.sh — runs run_ratio.sh ONE chain at a time, for each category/carrier whose run_cat.sh chain has
# finished (DONE in its run log), alongside cola_cat.sh (it is light: one generation + small YOLO runs).
R=/home/aliha/t23
cd $R
CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper"
left() { for c in $CATS; do for v in cattext catcolor; do
  [ -f logs/ratio_${v}_${c}.done ] || echo "$v $c"; done; done; }
while [ -n "$(left)" ]; do
  started=0
  while read v c; do
    grep -q DONE logs/${v}_${c}_run.log 2>/dev/null || continue
    echo "$(date '+%F %T') start $v $c" >> logs/cola_ratio.log
    if bash FIRSTPAPER-FINAL/prep/t23/run_ratio.sh $v $c > logs/ratio_${v}_${c}.log 2>&1 < /dev/null; then
      touch logs/ratio_${v}_${c}.done
    else
      echo "$(date '+%F %T') FAILED $v $c" >> logs/cola_ratio.log; touch logs/ratio_${v}_${c}.done
    fi
    started=1; break
  done < <(left)
  [ $started = 0 ] && sleep 600
done
echo "RATIO DONE" > logs/ratio_all.done
