PREFIX=1b-fp32-depth128-$iter DATA=train  bash embed-1b.sh
PREFIX=1b-fp32-depth128-$iter DATA=test  bash embed-1b.sh
bash test_lib.sh
bash test_saga.sh
