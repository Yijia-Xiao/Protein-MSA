# export iter=10
#PREFIX=trained-esm-fp32-depth128-$iter DATA=train  bash ./sh/embed-trained-esm.sh
#PREFIX=trained-esm-fp32-depth128-$iter DATA=test  bash ./sh/embed-trained-esm.sh
(python sklearn_ft.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale trained-esm) |& tee -a log/trained-esm-$iter-dp128-lib-job32.log
(python sklearn_ft.py --msa-depth 128 --solver saga --iter $iter --job-num 32 --model-scale trained-esm) |& tee -a log/trained-esm-$iter-dp128-lib-job32.log
