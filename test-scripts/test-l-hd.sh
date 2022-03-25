model_name=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd
depth=128
PREFIX=$model_name-depth$depth-$iter DATA=train  bash ./sh/embed-l-hd.sh
PREFIX=$model_name-depth$depth-$iter DATA=test  bash ./sh/embed-l-hd.sh
(python sklearn_ft.py --msa-depth $depth --solver liblinear --iter $iter --job-num 32 --model-scale $model_name) |& tee -a log/$model_name-$iter-dp$depth-lib-job32.log
(python sklearn_ft.py --msa-depth $depth --solver saga --iter $iter --job-num 32 --model-scale $model_name) |& tee -a log/$model_name-$iter-dp$depth-lib-job32.log
