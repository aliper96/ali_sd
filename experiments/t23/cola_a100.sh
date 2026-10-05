#!/bin/bash
# cola_a100.sh — queue the remaining generator trainings on the A100.
# Splits 0-2 are already training (3 jobs fill the GPU). Split 3 starts when split 0
# writes its final checkpoint, split 4 when split 1 does.
cd /home/aliha/t23
export PYTHONPATH=/home/aliha/aliaug/libs
run() {  # $1 = split to wait for, $2 = split to launch
  while [ ! -f gen_split_$1/checkpoints/model_10000.pkl ]; do sleep 120; done
  sleep 60
  python3 -u FIRSTPAPER-FINAL/prep/t23/train_gen.py --code /home/aliha/t23/ali_sd_public \
    --split /home/aliha/t23/aliaug_splits/split_$2 --out /home/aliha/t23/gen_split_$2 \
    --seed $2 --workers 2 > train_gen_split_$2.log 2>&1
}
run 0 3 &
run 1 4 &
wait
