PREFIX=corp-fp32-depth128-$iter DATA=train  bash ./sh/embed-corp.sh
PREFIX=corp-fp32-depth128-$iter DATA=test  bash ./sh/embed-corp.sh
#bash ./sh/test_lib.sh
(python sklearn_ft.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale corp) |& tee -a log/corp-$iter-dp128-lib-job32.log


# bash ./sh/test_saga.sh

# PREFIX=1b-fp32-depth128-$iter DATA=train  bash ./sh/embed-1b.sh
# PREFIX=1b-fp32-depth128-$iter DATA=test  bash ./sh/embed-1b.sh
# bash ./sh/test_lib.sh
# # bash ./sh/test_saga.sh
