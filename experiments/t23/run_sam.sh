#!/bin/bash
# run_sam.sh — E5 on the A100: SAM 3 as the automatic step of generate -> SAM -> human check.
# Prerequisites (done by the caller): sam3 code in /home/aliha/sam3_code, weights in the HF cache
# (/home/aliha/.cache/huggingface/hub/models--facebook--sam3), iopath installed.
# sam3 is put on PYTHONPATH WITHOUT installing its pinned deps (numpy==1.26 would downgrade the
# system numpy used by other projects on this VM).
R=/home/aliha/t23
cd $R
export PYTHONPATH=/home/aliha/sam3_code:/home/aliha/aliaug/libs HF_HUB_OFFLINE=1
mkdir -p results_sam
for k in 0 1 2 3 4; do
  # reference: SAM 3 on the REAL held-out defects (its ceiling on tile)
  python3 FIRSTPAPER-FINAL/prep/t23/sam_label.py --images aliaug_splits/split_$k/test_B \
    --masks aliaug_splits/split_$k/test_A --out results_sam/real_split$k.json > logs/sam_real_$k.log 2>&1
  for V in textfix colorfix; do
    D=judge_syn/${V}_heldout_split_$k
    [ -f $D/labels.json ] || continue
    python3 FIRSTPAPER-FINAL/prep/t23/sam_label.py --images $D --masks aliaug_splits/split_$k/test_A \
      --labels $D/labels.json --out results_sam/${V}_split$k.json > logs/sam_${V}_$k.log 2>&1
  done
done
echo "SAM DONE" > logs/sam_all.done
