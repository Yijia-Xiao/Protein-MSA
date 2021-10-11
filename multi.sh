#!/bin/bash

#set -u
set -x

# NCCL_DEBUG=info
NCCL_ENV="NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"

NNODES=8
# NNODES=2

# MP=1
# g_bs=256
# LAYERNUM=16
MP=1
g_bs=256
LAYERNUM=16
HIDDENSIZE=2048
HEAD=16

# MAX_TOKENS=16384
# MAX_TOKENS=65536
MAX_TOKENS=18432
MAX_ALIGNS=384
# MAX_LENGTH=1024
MAX_LENGTH=768

BATCHSIZE=1
DATE=release

WS=2000
ITER=200000

# g_bs=64
NAME=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd-${BATCHSIZE}mbs-${g_bs}gbs-${MP}mp-${MAX_TOKENS}tokens-${MAX_ALIGNS}aligns-${MAX_LENGTH}length-${WS}ws-${ITER}iter-${DATE}

MYPATH=$PWD

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_ADDR=node1
MASTER_PORT=7010

# DATA_PATH=/workspace/XA/XA_KING_text_document
# DATA_PATH=/workspace/XB/XB_KING_text_document
DATA_PATH=/workspace/DATA/UniRef50-xb-a2m-2018/UniRef50-xb-a2m-2018_text_document
DATA_PATH=/workspace/DATA/UniRef50-xc-a2m-2017/UniRef50-xc-a2m-2017_text_document
DATA_PATH=/workspace/DATA/UniRef50-xd-a2m-2018/UniRef50-xd-a2m-2018_text_document
# UniRef50-xc-a2m-2017
DATA_PATH=/workspace/DATA/TOTAL/TOTAL_text_document
# #OLD=1
# if [ -z $OLD ]; then
#   DATA_PATH=/workspace/XA/XA_KING_text_document
#   WS=16000
#   ITER=100000
# else
#   DATA_PATH=./msa_tools/fake_text_document
#   WS=10
#   ITER=100
# fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


CHECKPOINT_PATH=/workspace/ckpt/$DATE/$NAME
TB=$MYPATH/tb/$DATE/$NAME
LOG=$MYPATH/logs/$DATE/$NAME

mkdir -p $CHECKPOINT_PATH
mkdir -p $MYPATH/tb/$DATE/
mkdir -p $TB
mkdir -p $MYPATH/logs/$DATE/


# $MYPATH/msa_tools/msa_vocab.txt
# /dataset/ee84df8b/debug/pretrain/msa_tools/msa_vocab.txt
# python -m torch.distributed.launch $DISTRIBUTED_ARGS \

# ${NCCL_ENV}

NCCL_IB_DISABLE=0
NCCL_IB_GID_INDEX=3
NCCL_NET_GDR_LEVEL=0

(python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $MYPATH/pretrain_tape.py \
       --num-layers $LAYERNUM \
       --hidden-size $HIDDENSIZE \
       --num-attention-heads $HEAD \
       --micro-batch-size $BATCHSIZE \
       --global-batch-size ${g_bs} \
       --seq-length $MAX_LENGTH \
       --max-position-embeddings 1024 \
       --train-iters $ITER \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $MYPATH/msa_tools/msa_vocab.txt \
       --data-impl mmap \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --clip-grad 1.0 \
       --lr-warmup-iters $WS \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 100 \
       --max-tokens $MAX_TOKENS \
       --max-aligns $MAX_ALIGNS \
       --max-length $MAX_LENGTH \
       --tensor-model-parallel-size $MP \
       --fp16 \
       --no-scaled-masked-softmax-fusion \
       --tensorboard-dir $TB \
       --checkpoint-activations \
       --msa-shuffle \
       --add-msa-positional-embedding \
       --add-post-embedding-layernorm \
       --split 990,9,1
) |& tee -a $LOG
       # --checkpoint-num-layers 2 \


# run_cmd="${NCCL_ENV} python -m torch.distributed.launch $DISTRIBUTED_ARGS \
#        $MYPATH/pretrain_tape.py \
#        --num-layers $LAYERNUM \
#        --hidden-size $HIDDENSIZE \
#        --num-attention-heads $HEAD \
#        --micro-batch-size $BATCHSIZE \
#        --global-batch-size ${g_bs} \
#        --seq-length $MAX_LENGTH \
#        --max-position-embeddings $MAX_LENGTH \
#        --train-iters $ITER \
#        --save $CHECKPOINT_PATH \
#        --load $CHECKPOINT_PATH \
#        --data-path $DATA_PATH \
#        --vocab-file $MYPATH/msa_tools/msa_vocab.txt \
#        --data-impl mmap \
#        --distributed-backend nccl \
#        --lr 0.0001 \
#        --lr-decay-style linear \
#        --clip-grad 1.0 \
#        --lr-warmup-iters $WS \
#        --log-interval 1 \
#        --save-interval 500 \
#        --eval-interval 50 \
#        --eval-iters 100 \
#        --max-tokens $MAX_TOKENS \
#        --max-aligns $MAX_ALIGNS \
#        --max-length $MAX_LENGTH \
#        --tensor-model-parallel-size $MP \
#        --fp16 \
#        --no-scaled-masked-softmax-fusion \
#        --tensorboard-dir $TB \
#        --checkpoint-activations \
# "
# # ) |& tee -a $LOG

# echo "${run_cmd}"

# eval ${run_cmd}

# set +x

