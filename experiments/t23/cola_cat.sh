#!/bin/bash
# cola_cat.sh — queue for the per-category LoRA study (priority) and the remaining tile ablations.
# A job starts only when fewer than 3 chains are running (run_cat.sh or run_variant.sh, including the
# chains left from the previous queue) AND the GPU has >= 25 GB free, so no training can run out of memory.
R=/home/aliha/t23
S=/mnt/scratch/t23
cd $R
CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw toothbrush transistor wood zipper"
JOBS=()
for c in $CATS; do JOBS+=("cattext $c" "catcolor $c"); done
JOBS+=("abl_nogan 0" "abl_nolabel 0")
for k in 1 2; do for v in abl_noskip abl_noclip abl_nogan abl_nolabel; do JOBS+=("$v $k"); done; done

running() { ps -eo args | grep -cE '^bash FIRSTPAPER-FINAL/prep/t23/run_(cat|variant)\.sh'; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

for j in "${JOBS[@]}"; do
  set -- $j
  while [ "$(running)" -ge 3 ] || [ "$(free_mb)" -lt 25000 ]; do sleep 120; done
  case $1 in
    cat*) setsid bash FIRSTPAPER-FINAL/prep/t23/run_cat.sh $1 $2 > logs/${1}_${2}_run.log 2>&1 < /dev/null & ;;
    *)    setsid bash FIRSTPAPER-FINAL/prep/t23/run_variant.sh $1 $2 > logs/${1}_run_$2.log 2>&1 < /dev/null & ;;
  esac
  echo "$(date '+%F %T') started $j" >> logs/cola_cat.log
  sleep 300   # let the new training allocate its memory before the next check
done
while [ "$(running)" -gt 0 ]; do sleep 300; done
echo "CAT DONE" > logs/cat_all.done
