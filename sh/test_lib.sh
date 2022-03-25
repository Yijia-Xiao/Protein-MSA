# (python sklearn_ft.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale 1b) |& tee -a log/dmask-1b-$iter-dp128-lib-job32.log
(python sklearn_ft.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale 1b) |& tee -a log/1b-$iter-dp128-lib-job32.log
