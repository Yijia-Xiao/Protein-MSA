#!/bin/bash
set -x

NCCL_ENV="NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"

NNODES=1

MP=1
g_bs=256
LAYERNUM=12
HIDDENSIZE=1536
HEAD=12

MAX_TOKENS=32768
MAX_ALIGNS=1024
MAX_LENGTH=1536

BATCHSIZE=1
DATE=release

WS=8000
ITER=200000

# NAME=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd-${BATCHSIZE}mbs-${g_bs}gbs-${MP}mp-${MAX_ALIGNS}aligns-ength-${WS}ws-${ITER}iter-${DATE}
NAME=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd-${BATCHSIZE}mbs-${g_bs}gbs-${MP}mp-${MAX_TOKENS}tokens-${MAX_ALIGNS}aligns-${MAX_LENGTH}length-${WS}ws-${ITER}iter-${DATE}

MYPATH=$PWD

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_ADDR=node1
MASTER_PORT=7100

# DATA_PATH="0.173 /root/MSA/UniRef50-xa-a2m-2017/UniRef50-xa-a2m-2017_text_document 0.173 /root/MSA/UniRef50-xb-a2m-2018/UniRef50-xb-a2m-2018_text_document 0.173 /root/MSA/UniRef50-xc-a2m-2017/UniRef50-xc-a2m-2017_text_document 0.173 /root/MSA/UniRef50-xd-a2m-2018/UniRef50-xd-a2m-2018_text_document 0.167 /root/MSA/UniRef50-xe-a2m-2017/UniRef50-xe-a2m-2017_text_document 0.141 /root/MSA/UniRef50-xf-a2m-2018/UniRef50-xf-a2m-2018_text_document"
DATA_PATH=/root/DATA/TOTAL_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


CHECKPOINT_PATH=/workspace/ckpt/$DATE/$NAME
TB=$MYPATH/tb/$DATE/$NAME
LOG=$MYPATH/logs/$DATE/$NAME

mkdir -p $CHECKPOINT_PATH
mkdir -p $MYPATH/tb/$DATE/
mkdir -p $TB
mkdir -p $MYPATH/logs/$DATE/

NCCL_IB_DISABLE=0
NCCL_IB_GID_INDEX=3
NCCL_NET_GDR_LEVEL=0

# SE_ITER=100
SE_ITER=500

(python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $MYPATH/pretrain_msa.py \
       --num-layers $LAYERNUM \
       --hidden-size $HIDDENSIZE \
       --num-attention-heads $HEAD \
       --micro-batch-size $BATCHSIZE \
       --global-batch-size ${g_bs} \
       --seq-length $MAX_LENGTH \
       --max-position-embeddings $MAX_LENGTH \
       --max-msa-position-embeddings $MAX_ALIGNS \
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
       --save-interval $SE_ITER \
       --eval-interval $SE_ITER \
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
       --split 997,2,1 \
) |& tee -a $LOG


#       --split 996,3,1 \
#) |& tee -a $LOG



# --dynamic-mask 1
