#!/bin/bash

set -u
set -x


HIDDENSIZE=768
LAYERNUM=12
HEAD=12

MAX_ALIGNS=64
MAX_LENGTH=1026

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

ITER=10

DATA_PATH=/root/ProteinLM/pretrain/contact/data/megatron/xc_text_document
# if [ $DATA == xc ]; then
# elif [ $DATA == tr ]; then
#        DATA_PATH=/root/ProteinLM/pretrain/contact/data/megatron/msa_text_document
# fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ./pretrain_tape.py \
       --num-layers $LAYERNUM \
       --hidden-size $HIDDENSIZE \
       --num-attention-heads $HEAD \
       --micro-batch-size $BATCHSIZE \
       --global-batch-size ${g_bs} \
       --seq-length $MAX_LENGTH \
       --max-position-embeddings $MAX_LENGTH \
       --train-iters 1000 \
       --data-path $DATA_PATH \
       --vocab-file $MYPATH/msa_tools/msa_vocab.txt \
       --data-impl mmap \
       --distributed-backend nccl \
       --lr 0 \
       --log-interval 1 \
       --save-interval 5 \
       --eval-iters $ITER \
       --max-tokens $(($MAX_ALIGNS*$MAX_LENGTH)) \
       --max-aligns $MAX_ALIGNS \
       --max-length $MAX_LENGTH \
       --tensor-model-parallel-size $MP \
       --no-scaled-masked-softmax-fusion \
       --lr 0 \
       --mask-prob 0 \
       --save ./ckpt \
 
       # --eval-interval 1000 \

       # --use-cpu-initialization \

       # --attention-save \
       # --attention-name $PREFIX \
       # --finetune \
       # --attention-path /workspace/plt/attention/ \
       # --checkpoint-activations \