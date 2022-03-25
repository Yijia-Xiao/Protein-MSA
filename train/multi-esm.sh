#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NCCL_ENV="NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"

NNODES=1

MP=1
g_bs=168
g_bs=28
LAYERNUM=8
HIDDENSIZE=512
HEAD=8

MAX_TOKENS=16384
MAX_ALIGNS=128
MAX_LENGTH=1024

BATCHSIZE=1
DATE=release

WS=16000
ITER=250000

NAME=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd-${BATCHSIZE}mbs-${g_bs}gbs-${MP}mp-${MAX_TOKENS}tokens-${MAX_ALIGNS}aligns-${MAX_LENGTH}length-${WS}ws-${ITER}iter-${DATE}

MYPATH=/dataset/ee84df8b/Protein-MSA/ # $PWD

GPUS_PER_NODE=7
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_ADDR=localhost
MASTER_PORT=7100

# DATA_PATH="1 /dataset/ee84df8b/data/BIN/00_text_document 1 /dataset/ee84df8b/data/BIN/01_text_document 1 /dataset/ee84df8b/data/BIN/02_text_document 1 /dataset/ee84df8b/data/BIN/03_text_document 1 /dataset/ee84df8b/data/BIN/04_text_document 1 /dataset/ee84df8b/data/BIN/05_text_document 1 /dataset/ee84df8b/data/BIN/06_text_document 1 /dataset/ee84df8b/data/BIN/07_text_document 1 /dataset/ee84df8b/data/BIN/08_text_document 1 /dataset/ee84df8b/data/BIN/09_text_document 1 /dataset/ee84df8b/data/BIN/10_text_document 1 /dataset/ee84df8b/data/BIN/11_text_document 1 /dataset/ee84df8b/data/BIN/12_text_document 1 /dataset/ee84df8b/data/BIN/13_text_document 1 /dataset/ee84df8b/data/BIN/14_text_document 1 /dataset/ee84df8b/data/BIN/15_text_document 1 /dataset/ee84df8b/data/BIN/16_text_document 1 /dataset/ee84df8b/data/BIN/17_text_document 1 /dataset/ee84df8b/data/BIN/18_text_document 1 /dataset/ee84df8b/data/BIN/19_text_document 1 /dataset/ee84df8b/data/BIN/20_text_document 1 /dataset/ee84df8b/data/BIN/21_text_document 1 /dataset/ee84df8b/data/BIN/22_text_document 1 /dataset/ee84df8b/data/BIN/23_text_document 1 /dataset/ee84df8b/data/BIN/24_text_document"
# # DATA_PATH="1 /dataset/ee84df8b/data/BIN/25_text_document 1 /dataset/ee84df8b/data/BIN/26_text_document 1 /dataset/ee84df8b/data/BIN/27_text_document 1 /dataset/ee84df8b/data/BIN/28_text_document 1 /dataset/ee84df8b/data/BIN/29_text_document 1 /dataset/ee84df8b/data/BIN/30_text_document 1 /dataset/ee84df8b/data/BIN/31_text_document 1 /dataset/ee84df8b/data/BIN/32_text_document 1 /dataset/ee84df8b/data/BIN/33_text_document 1 /dataset/ee84df8b/data/BIN/34_text_document 1 /dataset/ee84df8b/data/BIN/35_text_document 1 /dataset/ee84df8b/data/BIN/36_text_document 1 /dataset/ee84df8b/data/BIN/37_text_document 1 /dataset/ee84df8b/data/BIN/38_text_document 1 /dataset/ee84df8b/data/BIN/39_text_document 1 /dataset/ee84df8b/data/BIN/40_text_document 1 /dataset/ee84df8b/data/BIN/41_text_document 1 /dataset/ee84df8b/data/BIN/42_text_document 1 /dataset/ee84df8b/data/BIN/43_text_document 1 /dataset/ee84df8b/data/BIN/44_text_document 1 /dataset/ee84df8b/data/BIN/45_text_document 1 /dataset/ee84df8b/data/BIN/46_text_document 1 /dataset/ee84df8b/data/BIN/47_text_document 1 /dataset/ee84df8b/data/BIN/48_text_document 1 /dataset/ee84df8b/data/BIN/49_text_document"

DATA_PATH="1 /dataset/ee84df8b/data/BIN/00_text_document 1 /dataset/ee84df8b/data/BIN/01_text_document 1 /dataset/ee84df8b/data/BIN/02_text_document 1 /dataset/ee84df8b/data/BIN/03_text_document 1 /dataset/ee84df8b/data/BIN/04_text_document 1 /dataset/ee84df8b/data/BIN/05_text_document 1 /dataset/ee84df8b/data/BIN/06_text_document 1 /dataset/ee84df8b/data/BIN/07_text_document 1 /dataset/ee84df8b/data/BIN/08_text_document 1 /dataset/ee84df8b/data/BIN/09_text_document 1 /dataset/ee84df8b/data/BIN/10_text_document 1 /dataset/ee84df8b/data/BIN/11_text_document 1 /dataset/ee84df8b/data/BIN/12_text_document 1 /dataset/ee84df8b/data/BIN/13_text_document 1 /dataset/ee84df8b/data/BIN/14_text_document 1 /dataset/ee84df8b/data/BIN/15_text_document 1 /dataset/ee84df8b/data/BIN/16_text_document 1 /dataset/ee84df8b/data/BIN/17_text_document 1 /dataset/ee84df8b/data/BIN/18_text_document 1 /dataset/ee84df8b/data/BIN/19_text_document 1 /dataset/ee84df8b/data/BIN/20_text_document 1 /dataset/ee84df8b/data/BIN/21_text_document 1 /dataset/ee84df8b/data/BIN/22_text_document 1 /dataset/ee84df8b/data/BIN/23_text_document 1 /dataset/ee84df8b/data/BIN/24_text_document 1 /dataset/ee84df8b/data/BIN/25_text_document 1 /dataset/ee84df8b/data/BIN/26_text_document 1 /dataset/ee84df8b/data/BIN/27_text_document 1 /dataset/ee84df8b/data/BIN/28_text_document 1 /dataset/ee84df8b/data/BIN/29_text_document 1 /dataset/ee84df8b/data/BIN/30_text_document 1 /dataset/ee84df8b/data/BIN/31_text_document 1 /dataset/ee84df8b/data/BIN/32_text_document 1 /dataset/ee84df8b/data/BIN/33_text_document 1 /dataset/ee84df8b/data/BIN/34_text_document 1 /dataset/ee84df8b/data/BIN/35_text_document 1 /dataset/ee84df8b/data/BIN/36_text_document 1 /dataset/ee84df8b/data/BIN/37_text_document 1 /dataset/ee84df8b/data/BIN/38_text_document 1 /dataset/ee84df8b/data/BIN/39_text_document 1 /dataset/ee84df8b/data/BIN/40_text_document 1 /dataset/ee84df8b/data/BIN/41_text_document 1 /dataset/ee84df8b/data/BIN/42_text_document 1 /dataset/ee84df8b/data/BIN/43_text_document 1 /dataset/ee84df8b/data/BIN/44_text_document 1 /dataset/ee84df8b/data/BIN/45_text_document 1 /dataset/ee84df8b/data/BIN/46_text_document 1 /dataset/ee84df8b/data/BIN/47_text_document 1 /dataset/ee84df8b/data/BIN/48_text_document 1 /dataset/ee84df8b/data/BIN/49_text_document"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


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

SE_ITER=1000

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
       --add-msa-positional-embedding \
       --add-post-embedding-layernorm \
       --split 39999,1,0 \
) |& tee -a $LOG
# --msa-shuffle \
# --split 996,3,1 \

