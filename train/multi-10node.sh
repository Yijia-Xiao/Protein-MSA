#!/bin/bash
set -x

NCCL_ENV="NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"

NNODES=10
#NNODES=8
NNODES=9

MP=1
g_bs=80
#g_bs=64
g_bs=72
LAYERNUM=6
HIDDENSIZE=768
HEAD=8

MAX_TOKENS=30208

MAX_ALIGNS=128
MAX_LENGTH=512
POS_EMBED=512

BATCHSIZE=1
DATE=release

WS=16000
ITER=250000
NAME=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd-${BATCHSIZE}mbs-${g_bs}gbs-${MP}mp-${MAX_TOKENS_ALL}tokens-${MAX_ALIGNS}aligns-${MAX_LENGTH}length-${WS}ws-${ITER}iter-${DATE}
NAME="768h-6l-8hd-1mbs-128gbs-1mp-tokens-128aligns-512length-16000ws-250000iter-release"

MYPATH=$PWD

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_ADDR=node11
MASTER_ADDR=node1
MASTER_PORT=7100

DATA_PATH="1 /dataset/ee84df8b/data/BIN/04_text_document 1 /dataset/ee84df8b/data/BIN/05_text_document 1 /dataset/ee84df8b/data/BIN/06_text_document 1 /dataset/ee84df8b/data/BIN/07_text_document 1 /dataset/ee84df8b/data/BIN/08_text_document 1 /dataset/ee84df8b/data/BIN/09_text_document 1 /dataset/ee84df8b/data/BIN/14_text_document 1 /dataset/ee84df8b/data/BIN/15_text_document 1 /dataset/ee84df8b/data/BIN/16_text_document 1 /dataset/ee84df8b/data/BIN/17_text_document 1 /dataset/ee84df8b/data/BIN/18_text_document 1 /dataset/ee84df8b/data/BIN/19_text_document 1 /dataset/ee84df8b/data/BIN/20_text_document 1 /dataset/ee84df8b/data/BIN/21_text_document 1 /dataset/ee84df8b/data/BIN/22_text_document 1 /dataset/ee84df8b/data/BIN/23_text_document 1 /dataset/ee84df8b/data/BIN/24_text_document 1 /dataset/ee84df8b/data/BIN/25_text_document 1 /dataset/ee84df8b/data/BIN/26_text_document 1 /dataset/ee84df8b/data/BIN/27_text_document 1 /dataset/ee84df8b/data/BIN/28_text_document 1 /dataset/ee84df8b/data/BIN/29_text_document 1 /dataset/ee84df8b/data/BIN/30_text_document 1 /dataset/ee84df8b/data/BIN/31_text_document 1 /dataset/ee84df8b/data/BIN/32_text_document 1 /dataset/ee84df8b/data/BIN/33_text_document 1 /dataset/ee84df8b/data/BIN/34_text_document 1 /dataset/ee84df8b/data/BIN/35_text_document 1 /dataset/ee84df8b/data/BIN/36_text_document 1 /dataset/ee84df8b/data/BIN/37_text_document 1 /dataset/ee84df8b/data/BIN/38_text_document 1 /dataset/ee84df8b/data/BIN/39_text_document 1 /dataset/ee84df8b/data/BIN/40_text_document 1 /dataset/ee84df8b/data/BIN/41_text_document 1 /dataset/ee84df8b/data/BIN/42_text_document 1 /dataset/ee84df8b/data/BIN/43_text_document 1 /dataset/ee84df8b/data/BIN/44_text_document 1 /dataset/ee84df8b/data/BIN/45_text_document 1 /dataset/ee84df8b/data/BIN/46_text_document 1 /dataset/ee84df8b/data/BIN/47_text_document 1 /dataset/ee84df8b/data/BIN/48_text_document 1 /dataset/ee84df8b/data/BIN/49_text_document 1 /dataset/ee84df8b/data/BIN/50_text_document 1 /dataset/ee84df8b/data/BIN/51_text_document 1 /dataset/ee84df8b/data/BIN/52_text_document 1 /dataset/ee84df8b/data/BIN/53_text_document 1 /dataset/ee84df8b/data/BIN/54_text_document 1 /dataset/ee84df8b/data/BIN/55_text_document 1 /dataset/ee84df8b/data/BIN/56_text_document 1 /dataset/ee84df8b/data/BIN/57_text_document 1 /dataset/ee84df8b/data/BIN/58_text_document 1 /dataset/ee84df8b/data/BIN/59_text_document 1 /dataset/ee84df8b/data/BIN/60_text_document 1 /dataset/ee84df8b/data/BIN/61_text_document 1 /dataset/ee84df8b/data/BIN/62_text_document 1 /dataset/ee84df8b/data/BIN/63_text_document 1 /dataset/ee84df8b/data/BIN/64_text_document 1 /dataset/ee84df8b/data/BIN/65_text_document 1 /dataset/ee84df8b/data/BIN/66_text_document 1 /dataset/ee84df8b/data/BIN/67_text_document 1 /dataset/ee84df8b/data/BIN/68_text_document 1 /dataset/ee84df8b/data/BIN/69_text_document 1 /dataset/ee84df8b/data/BIN/70_text_document 1 /dataset/ee84df8b/data/BIN/71_text_document 1 /dataset/ee84df8b/data/BIN/72_text_document 1 /dataset/ee84df8b/data/BIN/73_text_document 1 /dataset/ee84df8b/data/BIN/74_text_document 1 /dataset/ee84df8b/data/BIN/75_text_document 1 /dataset/ee84df8b/data/BIN/76_text_document 1 /dataset/ee84df8b/data/BIN/77_text_document 1 /dataset/ee84df8b/data/BIN/78_text_document 1 /dataset/ee84df8b/data/BIN/79_text_document 1 /dataset/ee84df8b/data/BIN/80_text_document 1 /dataset/ee84df8b/data/BIN/81_text_document 1 /dataset/ee84df8b/data/BIN/82_text_document 1 /dataset/ee84df8b/data/BIN/83_text_document 1 /dataset/ee84df8b/data/BIN/84_text_document 1 /dataset/ee84df8b/data/BIN/85_text_document 1 /dataset/ee84df8b/data/BIN/86_text_document 1 /dataset/ee84df8b/data/BIN/87_text_document 1 /dataset/ee84df8b/data/BIN/88_text_document 1 /dataset/ee84df8b/data/BIN/89_text_document 1 /dataset/ee84df8b/data/BIN/90_text_document 1 /dataset/ee84df8b/data/BIN/91_text_document 1 /dataset/ee84df8b/data/BIN/92_text_document 1 /dataset/ee84df8b/data/BIN/93_text_document 1 /dataset/ee84df8b/data/BIN/94_text_document 1 /dataset/ee84df8b/data/BIN/95_text_document 1 /dataset/ee84df8b/data/BIN/96_text_document 1 /dataset/ee84df8b/data/BIN/97_text_document 1 /dataset/ee84df8b/data/BIN/98_text_document 1 /dataset/ee84df8b/data/BIN/99_text_document"
# DATA_PATH="1 /dataset/ee84df8b/data/BIN/04_text_document"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


CHECKPOINT_PATH=/dataset/ee84df8b/workspace/exp/$NAME
TB=$MYPATH/tb/$DATE/$NAME
LOG=$MYPATH/logs/$DATE/$NAME

mkdir -p $CHECKPOINT_PATH
mkdir -p $MYPATH/tb/$DATE/
mkdir -p $TB
mkdir -p $MYPATH/logs/$DATE/

NCCL_IB_DISABLE=0
NCCL_IB_GID_INDEX=3
NCCL_NET_GDR_LEVEL=0

SE_ITER=500

(python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $MYPATH/pretrain_msa.py \
       --num-layers $LAYERNUM \
       --hidden-size $HIDDENSIZE \
       --num-attention-heads $HEAD \
       --micro-batch-size $BATCHSIZE \
       --global-batch-size ${g_bs} \
       --seq-length $MAX_LENGTH \
       --max-position-embeddings $POS_EMBED \
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
       --no-scaled-masked-softmax-fusion \
       --tensorboard-dir $TB \
       --add-msa-positional-embedding \
       --add-post-embedding-layernorm \
       --split 999,1,0 \
       --override-lr-scheduler \
) |& tee -a $LOG
