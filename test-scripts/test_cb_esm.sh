export iter=10
PREFIX=cb-test-esm-fp32-depth128-$iter DATA=train  bash ./sh/embed-esm-cb-test.sh
PREFIX=cb-test-esm-fp32-depth128-$iter DATA=test  bash ./sh/embed-esm-cb-test.sh
PREFIX=cb-test-esm-fp32-depth128-$iter DATA=valid  bash ./sh/embed-esm-cb-test.sh
(python test_cb513.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale esm) |& tee -a log/cb-esm-testset-$iter-dp128-lib-job32.log
(python valid_cb513.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale esm) |& tee -a log/cb-esm-validset-$iter-dp128-lib-job32.log
