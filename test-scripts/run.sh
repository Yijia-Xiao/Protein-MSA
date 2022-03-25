export MASTER_PORT=$(($CUDA+7010));

inter=8
for (( i=0;$i<$inter;i=`expr $i+1` ));
do
    # echo CUDA_VISIBLE_DEVICES=$CUDA iter=$((217500+$CUDA*$inter*500+$i*500)) bash test-corp.sh;
    # CUDA_VISIBLE_DEVICES=$CUDA
    iter=$((239500+$CUDA*$inter*500+$i*500)) bash test-corp.sh;
    # echo CUDA_VISIBLE_DEVICES=$CUDA iter=$((180500+$CUDA*$inter*500+$i*500)) bash test-corp.sh;
    # CUDA_VISIBLE_DEVICES=$CUDA iter=$((159500+$CUDA*$inter*500+$i*500)) bash test-corp.sh;
done;
# 159000 -> 180000 -> 217000 -> 239500 -> 250000
