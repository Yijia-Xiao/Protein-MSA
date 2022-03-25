export iter=10
PREFIX=cb-test-esm-fp32-depth128-$iter DATA=train  bash ./sh/embed-esm-cb-test.sh
#PREFIX=cb-test-esm-fp32-depth128-$iter DATA=test  bash ./sh/embed-esm-cb-test.sh
#PREFIX=cb-test-esm-fp32-depth128-$iter DATA=valid  bash ./sh/embed-esm-cb-test.sh
python test_cb513.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale esm
#python valid_cb513.py --msa-depth 128 --solver liblinear --iter $iter --job-num 32 --model-scale esm
