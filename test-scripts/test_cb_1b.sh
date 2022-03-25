PREFIX=cb-test-1b-fp32-depth128-$iter DATA=train  bash ./sh/embed-1b-cb-test.sh
PREFIX=cb-test-1b-fp32-depth128-$iter DATA=test  bash ./sh/embed-1b-cb-test.sh
PREFIX=cb-test-1b-fp32-depth128-$iter DATA=valid  bash ./sh/embed-1b-cb-test.sh
(python test_cb513.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale 1b) |& tee -a log/cb-1b-testset-$iter-dp128-lib-job32.log
(python valid_cb513.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale 1b) |& tee -a log/cb-1b-validset-$iter-dp128-lib-job32.log
