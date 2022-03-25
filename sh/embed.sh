#!/bin/bash

set -u
set -x

HIDDENSIZE=768
LAYERNUM=12
HEAD=12

# MAX_ALIGNS=256
# MAX_ALIGNS=64
MAX_ALIGNS=128
# MAX_LENGTH=1026
MAX_LENGTH=1024

if [ $CKPT == esm ]; then
       ckpt_path=./dump
elif [ $CKPT == 100m ]; then
       ckpt_path=/workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release
fi

BATCHSIZE=1
MP=1

g_bs=1

MYPATH=$PWD

GPUS_PER_NODE=1
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7008

if [ $DATA == train ]; then
       DATA_PATH=/dataset/ee84df8b/Protein-MSA/data/iupac/train_text_document
       EVAL_ITER=21
elif [ $DATA == test ]; then
       DATA_PATH=/dataset/ee84df8b/Protein-MSA/data/iupac/test_text_document
       EVAL_ITER=129
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ $CKPT == esm ]; then
       cmd="""python -m torch.distributed.launch $DISTRIBUTED_ARGS \
              ./pretrain_tape.py \
              --num-layers $LAYERNUM \
              --hidden-size $HIDDENSIZE \
              --num-attention-heads $HEAD \
              --micro-batch-size $BATCHSIZE \
              --global-batch-size ${g_bs} \
              --seq-length 1025 \
              --max-position-embeddings 1026 \
              --train-iters 1 \
              --data-path $DATA_PATH \
              --vocab-file $MYPATH/msa_tools/iupac-msa.txt \
              --data-impl mmap \
              --distributed-backend nccl \
              --lr 0 \
              --log-interval 1 \
              --save-interval 2000 \
              --eval-interval 1 \
              --eval-iters $EVAL_ITER \
              --max-tokens $(($MAX_ALIGNS*$MAX_LENGTH)) \
              --max-aligns $MAX_ALIGNS \
              --max-length $MAX_LENGTH \
              --tensor-model-parallel-size $MP \
              --no-scaled-masked-softmax-fusion \
              --override-lr-scheduler \
              --mask-prob 0 \
              --split 0,0,1 \
              --attention-save \
              --attention-name esm \
              --finetune \
              --attention-path /workspace/plt/attention/ \
              --load $ckpt_path \
              --attention-dropout 0 \
              --hidden-dropout 0 \
              --add-msa-positional-embedding \
              --add-post-embedding-layernorm \
       """
elif [ $CKPT == 100m ]; then
       cmd="""python -m torch.distributed.launch $DISTRIBUTED_ARGS \
              ./pretrain_tape.py \
              --num-layers $LAYERNUM \
              --hidden-size $HIDDENSIZE \
              --num-attention-heads $HEAD \
              --micro-batch-size $BATCHSIZE \
              --global-batch-size ${g_bs} \
              --seq-length $MAX_LENGTH \
              --max-position-embeddings $MAX_LENGTH \
              --train-iters 1 \
              --data-path $DATA_PATH \
              --vocab-file $MYPATH/msa_tools/msa_vocab.txt \
              --data-impl mmap \
              --distributed-backend nccl \
              --lr 0 \
              --log-interval 1 \
              --save-interval 2000 \
              --eval-interval 1 \
              --eval-iters $EVAL_ITER \
              --max-tokens $(($MAX_ALIGNS*$MAX_LENGTH)) \
              --max-aligns $MAX_ALIGNS \
              --max-length $MAX_LENGTH \
              --tensor-model-parallel-size $MP \
              --no-scaled-masked-softmax-fusion \
              --override-lr-scheduler \
              --mask-prob 0 \
              --split 0,0,1 \
              --attention-save \
              --attention-name 100m \
              --finetune \
              --attention-path /workspace/plt/attention/ \
              --load $ckpt_path \
              --attention-dropout 0 \
              --hidden-dropout 0 \
              --add-msa-positional-embedding \
              --add-post-embedding-layernorm \
              --eval-load-iter $iter \
       """
fi


eval $cmd
