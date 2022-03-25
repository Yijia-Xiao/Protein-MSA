#!/bin/bash

set -u
set -x

HIDDENSIZE=768
LAYERNUM=12
HEAD=12

# MAX_ALIGNS=256
# MAX_ALIGNS=64
MAX_ALIGNS=128
MAX_LENGTH=1026

ckpt_path=./dump
# ckpt_path=/workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release

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


# if [ $DATA == xc ]; then
#        DATA_PATH=/root/ProteinLM/pretrain/contact/data/megatron/xc_text_document
# elif [ $DATA == tr ]; then
#        DATA_PATH=/root/ProteinLM/pretrain/contact/data/megatron/msa_text_document
# fi
if [ $DATA == train ]; then
       DATA_PATH=/dataset/ee84df8b/release/ProteinLM/pretrain/data/iupac/train_text_document
       EVAL_ITER=21
elif [ $DATA == test ]; then
       DATA_PATH=/dataset/ee84df8b/release/ProteinLM/pretrain/data/iupac/test_text_document
       EVAL_ITER=129
fi


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
       --attention-name $PREFIX \
       --finetune \
       --attention-path /workspace/plt/attention/ \
       --load $ckpt_path \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --add-msa-positional-embedding \
       --add-post-embedding-layernorm \
       --fake-input \
       # --checkpoint-activations \

       # --fp16 \
