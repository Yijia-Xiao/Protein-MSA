#!/bin/bash
set -x

NCCL_ENV="NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"

NNODES=8
NNODES=12 # from 70500
NNODES=8 # from 75500
NNODES=9 # from 24800
NNODES=6

MP=1
g_bs=256 # to 70500
g_bs=192 # from 70500, to 12 node2
g_bs=256 # from 75500
g_bs=216 # from 24800
g_bs=480

LAYERNUM=8
HIDDENSIZE=2304
HEAD=12

# MAX_TOKENS_ORI=32768 # from 0 to 500 iters
MAX_TOKENS_ALL=30720 # from 500 to 12500
MAX_TOKENS=30208

MAX_ALIGNS=1024
MAX_LENGTH=1536
POS_EMBED=2048

BATCHSIZE=1
DATE=release

WS=16000
ITER=250000
ITER_OLD=200000
# NAME=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd-${BATCHSIZE}mbs-${g_bs}gbs-${MP}mp-${MAX_ALIGNS}aligns-ength-${WS}ws-${ITER}iter-${DATE}
NAME=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd-${BATCHSIZE}mbs-${g_bs}gbs-${MP}mp-${MAX_TOKENS_ALL}tokens-${MAX_ALIGNS}aligns-${MAX_LENGTH}length-${WS}ws-${ITER_OLD}iter-${DATE}
# NAME=${HIDDENSIZE}h-${LAYERNUM}l-${HEAD}hd-${BATCHSIZE}mbs-${g_bs}gbs-${MP}mp-${MAX_TOKENS_ORI}tokens-${MAX_ALIGNS}aligns-${MAX_LENGTH}length-${WS}ws-${ITER}iter-${DATE}
NAME="2304h-8l-12hd-1mbs-256gbs-1mp-30720tokens-1024aligns-1536length-16000ws-200000iter-release"


MYPATH=$PWD

GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_ADDR=node1
MASTER_PORT=7100

# DATA_PATH="0.173 /root/MSA/UniRef50-xa-a2m-2017/UniRef50-xa-a2m-2017_text_document 0.173 /root/MSA/UniRef50-xb-a2m-2018/UniRef50-xb-a2m-2018_text_document 0.173 /root/MSA/UniRef50-xc-a2m-2017/UniRef50-xc-a2m-2017_text_document 0.173 /root/MSA/UniRef50-xd-a2m-2018/UniRef50-xd-a2m-2018_text_document 0.167 /root/MSA/UniRef50-xe-a2m-2017/UniRef50-xe-a2m-2017_text_document 0.141 /root/MSA/UniRef50-xf-a2m-2018/UniRef50-xf-a2m-2018_text_document"
# DATA_PATH=/root/DATA/TOTAL_text_document # from 0 to 36500
# DATA_PATH=/dataset/ee84df8b/data/BIN/MSA_BL128_4_text_document # performance not good, ckpt moved to 
# DATA_PATH="0.5 /dataset/ee84df8b/data/BIN/MSA_AB384BL512_0_0_text_document 0.5 /dataset/ee84df8b/data/BIN/MSA_AB384BL512_0_1_text_document"

# DATA_PATH=/root/DATA/TOTAL_text_document # from 36500 on to 46000
# DATA_PATH="0.125 /dataset/ee84df8b/data/BIN/00_text_document 0.125 /dataset/ee84df8b/data/BIN/01_text_document 0.125 /dataset/ee84df8b/data/BIN/02_text_document 0.125 /dataset/ee84df8b/data/BIN/03_text_document 0.125 /dataset/ee84df8b/data/BIN/10_text_document 0.125 /dataset/ee84df8b/data/BIN/11_text_document 0.125 /dataset/ee84df8b/data/BIN/12_text_document 0.125 /dataset/ee84df8b/data/BIN/13_text_document" # from 46000 to 61500: split: 9999,1,0

# DATA_PATH="1 /dataset/ee84df8b/data/BIN/04_text_document 1 /dataset/ee84df8b/data/BIN/05_text_document 1 /dataset/ee84df8b/data/BIN/06_text_document 1 /dataset/ee84df8b/data/BIN/07_text_document 1 /dataset/ee84df8b/data/BIN/08_text_document 1 /dataset/ee84df8b/data/BIN/09_text_document 1 /dataset/ee84df8b/data/BIN/14_text_document 1 /dataset/ee84df8b/data/BIN/15_text_document 1 /dataset/ee84df8b/data/BIN/16_text_document 1 /dataset/ee84df8b/data/BIN/17_text_document 1 /dataset/ee84df8b/data/BIN/18_text_document 1 /dataset/ee84df8b/data/BIN/19_text_document 1 /dataset/ee84df8b/data/BIN/20_text_document 1 /dataset/ee84df8b/data/BIN/21_text_document 1 /dataset/ee84df8b/data/BIN/22_text_document 1 /dataset/ee84df8b/data/BIN/23_text_document 1 /dataset/ee84df8b/data/BIN/24_text_document"

# from 61500: change split from 9999,1,0 to 39999,1,0


DATA_PATH="1 /dataset/ee84df8b/data/BIN/04_text_document 1 /dataset/ee84df8b/data/BIN/05_text_document 1 /dataset/ee84df8b/data/BIN/06_text_document 1 /dataset/ee84df8b/data/BIN/07_text_document 1 /dataset/ee84df8b/data/BIN/08_text_document 1 /dataset/ee84df8b/data/BIN/09_text_document 1 /dataset/ee84df8b/data/BIN/14_text_document 1 /dataset/ee84df8b/data/BIN/15_text_document 1 /dataset/ee84df8b/data/BIN/16_text_document 1 /dataset/ee84df8b/data/BIN/17_text_document 1 /dataset/ee84df8b/data/BIN/18_text_document 1 /dataset/ee84df8b/data/BIN/19_text_document 1 /dataset/ee84df8b/data/BIN/20_text_document 1 /dataset/ee84df8b/data/BIN/21_text_document 1 /dataset/ee84df8b/data/BIN/22_text_document 1 /dataset/ee84df8b/data/BIN/23_text_document 1 /dataset/ee84df8b/data/BIN/24_text_document 1 /dataset/ee84df8b/data/BIN/25_text_document 1 /dataset/ee84df8b/data/BIN/26_text_document 1 /dataset/ee84df8b/data/BIN/27_text_document 1 /dataset/ee84df8b/data/BIN/28_text_document 1 /dataset/ee84df8b/data/BIN/29_text_document 1 /dataset/ee84df8b/data/BIN/30_text_document 1 /dataset/ee84df8b/data/BIN/31_text_document 1 /dataset/ee84df8b/data/BIN/32_text_document 1 /dataset/ee84df8b/data/BIN/33_text_document 1 /dataset/ee84df8b/data/BIN/34_text_document 1 /dataset/ee84df8b/data/BIN/35_text_document 1 /dataset/ee84df8b/data/BIN/36_text_document 1 /dataset/ee84df8b/data/BIN/37_text_document 1 /dataset/ee84df8b/data/BIN/38_text_document 1 /dataset/ee84df8b/data/BIN/39_text_document 1 /dataset/ee84df8b/data/BIN/40_text_document 1 /dataset/ee84df8b/data/BIN/41_text_document 1 /dataset/ee84df8b/data/BIN/42_text_document 1 /dataset/ee84df8b/data/BIN/43_text_document 1 /dataset/ee84df8b/data/BIN/44_text_document 1 /dataset/ee84df8b/data/BIN/45_text_document 1 /dataset/ee84df8b/data/BIN/46_text_document 1 /dataset/ee84df8b/data/BIN/47_text_document 1 /dataset/ee84df8b/data/BIN/48_text_document 1 /dataset/ee84df8b/data/BIN/49_text_document 1 /dataset/ee84df8b/data/BIN/50_text_document 1 /dataset/ee84df8b/data/BIN/51_text_document 1 /dataset/ee84df8b/data/BIN/52_text_document 1 /dataset/ee84df8b/data/BIN/53_text_document 1 /dataset/ee84df8b/data/BIN/54_text_document 1 /dataset/ee84df8b/data/BIN/55_text_document 1 /dataset/ee84df8b/data/BIN/56_text_document 1 /dataset/ee84df8b/data/BIN/57_text_document 1 /dataset/ee84df8b/data/BIN/58_text_document 1 /dataset/ee84df8b/data/BIN/59_text_document 1 /dataset/ee84df8b/data/BIN/60_text_document 1 /dataset/ee84df8b/data/BIN/61_text_document 1 /dataset/ee84df8b/data/BIN/62_text_document 1 /dataset/ee84df8b/data/BIN/63_text_document 1 /dataset/ee84df8b/data/BIN/64_text_document 1 /dataset/ee84df8b/data/BIN/65_text_document 1 /dataset/ee84df8b/data/BIN/66_text_document 1 /dataset/ee84df8b/data/BIN/67_text_document 1 /dataset/ee84df8b/data/BIN/68_text_document 1 /dataset/ee84df8b/data/BIN/69_text_document 1 /dataset/ee84df8b/data/BIN/70_text_document 1 /dataset/ee84df8b/data/BIN/71_text_document 1 /dataset/ee84df8b/data/BIN/72_text_document 1 /dataset/ee84df8b/data/BIN/73_text_document 1 /dataset/ee84df8b/data/BIN/74_text_document 1 /dataset/ee84df8b/data/BIN/75_text_document 1 /dataset/ee84df8b/data/BIN/76_text_document 1 /dataset/ee84df8b/data/BIN/77_text_document 1 /dataset/ee84df8b/data/BIN/78_text_document 1 /dataset/ee84df8b/data/BIN/79_text_document 1 /dataset/ee84df8b/data/BIN/80_text_document 1 /dataset/ee84df8b/data/BIN/81_text_document 1 /dataset/ee84df8b/data/BIN/82_text_document 1 /dataset/ee84df8b/data/BIN/83_text_document 1 /dataset/ee84df8b/data/BIN/84_text_document 1 /dataset/ee84df8b/data/BIN/85_text_document 1 /dataset/ee84df8b/data/BIN/86_text_document 1 /dataset/ee84df8b/data/BIN/87_text_document 1 /dataset/ee84df8b/data/BIN/88_text_document 1 /dataset/ee84df8b/data/BIN/89_text_document 1 /dataset/ee84df8b/data/BIN/90_text_document 1 /dataset/ee84df8b/data/BIN/91_text_document 1 /dataset/ee84df8b/data/BIN/92_text_document 1 /dataset/ee84df8b/data/BIN/93_text_document 1 /dataset/ee84df8b/data/BIN/94_text_document 1 /dataset/ee84df8b/data/BIN/95_text_document 1 /dataset/ee84df8b/data/BIN/96_text_document 1 /dataset/ee84df8b/data/BIN/97_text_document 1 /dataset/ee84df8b/data/BIN/98_text_document 1 /dataset/ee84df8b/data/BIN/99_text_document"


DATA_PATH="/dataset/ee84df8b/workspace/DATA/TOTAL/TOTAL_text_document" # from 155500 iters

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


# CHECKPOINT_PATH=/workspace/ckpt/$DATE/$NAME # from 0 tp 70500
CHECKPOINT_PATH="/dataset/ee84df8b/workspace/ckpt/release/2304h-8l-12hd-1mbs-256gbs-1mp-30720tokens-1024aligns-1536length-16000ws-200000iter-release/" # from 70500 on
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
# SE_ITER=1000

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
       --fp16 \
       --no-scaled-masked-softmax-fusion \
       --tensorboard-dir $TB \
       --checkpoint-activations \
       --add-msa-positional-embedding \
       --add-post-embedding-layernorm \
       --split 19999,1,0 \
       --override-lr-scheduler \
) |& tee -a $LOG

#       --split 29999,1,0 \ # 14 nodes
 
# --msa-shuffle \

#       --split 9999,1,0 \
#)

#       --split 996,3,1 \
#) |& tee -a $LOG

# --dynamic-mask 1
