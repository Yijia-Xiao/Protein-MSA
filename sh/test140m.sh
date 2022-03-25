export MAX_ALIGNS=128
CKPT=140m PREFIX=140m-fp32-depth$MAX_ALIGNS-$iter DATA=train  bash embed-1b-140m-60m.sh
CKPT=140m PREFIX=140m-fp32-depth$MAX_ALIGNS-$iter DATA=test  bash embed-1b-140m-60m.sh
(python sklearn_ft.py --msa-depth $MAX_ALIGNS --solver liblinear --iter $iter --job-num 32 --model-scale 140m) |& tee -a log/140m-$iter-dp128-lib-job32.log
(python sklearn_ft.py --msa-depth $MAX_ALIGNS --solver saga --iter $iter --job-num 32 --model-scale 140m) |& tee -a log/140m-$iter-dp128-saga-job32.log


# prev test
# CKPT=140m PREFIX=140m-fp32-depth128-$iter DATA=train  bash embed-1b-140m.sh
# CKPT=140m PREFIX=140m-fp32-depth128-$iter DATA=test  bash embed-1b-140m.sh
# (python sklearn_ft.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale 140m) |& tee -a log/140m-$iter-dp128-lib-job32.log
# (python sklearn_ft.py --msa-depth 128 --solver saga --iter $iter --job-num 32 --model-scale 140m) |& tee -a log/140m-$iter-dp128-saga-job32.log
