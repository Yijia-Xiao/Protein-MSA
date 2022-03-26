import os
import sys
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple
import torch
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
import logging
import argparse
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:\t%(message)s"
)

DATA_ROOT = './data/contact-data/'
parser = argparse.ArgumentParser(description="Parser for sklearn solver choice, MSA depth, and checkpoint etc.")
parser.add_argument(
    "--msa-depth", type=int, help="the depth of MSA"
)
parser.add_argument(
    "--solver", type=str, choices=['liblinear', 'saga'], help="skelarn solver used for fitting logistic regression model, choose from ['liblinear', 'saga']"
)
parser.add_argument(
    "--iter", type=int, help="take the checkpoint @iter"
)
parser.add_argument(
    "--job-num", type=int, default=16, help="the number of jobs in proba prediction"
)
parser.add_argument(
    "--model-scale", type=str, help="model scale"
    # "--model-scale", type=str, choices=['1b', 'esm', '100m', '140m', '60m', 'trained-esm'], help="model scale"
)

args = parser.parse_args()
logging.info(args)
msa_depth = args.msa_depth
sklearn_solver = args.solver
ckpt_iter = args.iter
job_num = args.job_num
model_scale = args.model_scale
# if model_scale == '1b':
if model_scale == '1b' or model_scale == 'esm' or model_scale == 'trained-esm' or model_scale == '768h-8l-6hd':
    max_len = 768
elif model_scale == '100m' or model_scale == '140m' or model_scale == '60m':
    max_len = 1024

alphabet_str = 'ARNDCQEGHILKMFPSTWYV-'
id_to_char = dict()

for i in range(len(alphabet_str)):
    id_to_char[i] = alphabet_str[i]


def map_id_to_token(token_id):
    return id_to_char[token_id]


def prepare_megatron():
    def process_trRosetta():
        os.system(f'rm {DATA_ROOT}/megatron/train.json')
        ret_data = []
        npz_path = f"{DATA_ROOT}/train20/"
        files = os.listdir(npz_path)
        for f in files:
            abs_path = os.path.join(npz_path, f)
            data = np.load(abs_path)
            msa_ids = data['msa']
            xyzCa = data['xyzca']
            msa_ids_2D_list = msa_ids.tolist()
            msa = []
            for seq in msa_ids_2D_list:
                msa_str = ''.join(list(map(map_id_to_token, seq)))
                msa.append(msa_str)
            contact = np.less(squareform(pdist(xyzCa)), 8.0).astype(np.int64)
            sample = {'name': f, 'msa': msa, 'contact': contact}
            with open(f'{DATA_ROOT}/megatron/train.json', 'a+') as f:
                one_seq = sample['msa'][0] + '|' + ''.join(sample['msa'][1:])
                one_seq = ' '.join(one_seq)
                f.write('{"text": ' + '"{}"}}'.format(one_seq) + "\n")
            ret_data.append(sample)
        return np.array(ret_data, dtype=object)

    def process_one_sample(file_name):
        msa_path = f'{DATA_ROOT}/CAMEO-trRosettaA2M'
        label_path = f'{DATA_ROOT}/CAMEO-GroundTruth'
        msa = open(os.path.join(msa_path, file_name + '.a2m'), 'r').read().splitlines()  # .readlines()
        with open(os.path.join(label_path, file_name + '.native.pkl'), 'rb') as f:
            labels = pickle.load(f, encoding="bytes")

        assert msa[0].strip() == labels[b'sequence'].decode()
        # an entry of <0 indicates an invalid distance.
        dist_mat = labels[b'atomDistMatrix'][b'CbCb']
        seq_len = len(dist_mat)
        binary_labels = torch.zeros((seq_len, seq_len), dtype=torch.float).tolist()
        for row in range(seq_len):
            for col in range(seq_len):
                if dist_mat[row][col] >= 0:
                    if dist_mat[row][col] < 8:
                        binary_labels[row][col] = 1
                else:
                    binary_labels[row][col] = -1
        return {
            'name': file_name,
            'msa': msa,
            'binary_labels': binary_labels,
        }

    def process_CAMEO():
        trRosetta_data = []
        label_path = f'{DATA_ROOT}/CAMEO-GroundTruth'
        names = [i.split('.')[0] for i in os.listdir(label_path)]
        for name in names:
            data = process_one_sample(name)
            # logging.info(data['msa'])
            with open(f'{DATA_ROOT}/megatron/test.json', 'a+') as f:
                one_seq = data['msa'][0] + '|' + ''.join(data['msa'][1:])
                one_seq = ' '.join(one_seq)
                f.write('{"text": ' + '"{}"}}'.format(one_seq) + "\n")

            trRosetta_data.append(data)

        return np.array(trRosetta_data, dtype=object)

    train = process_trRosetta()
    logging.info(len(train))
    np.save(f'{DATA_ROOT}/megatron/train_dataset', train, allow_pickle=True)

    test = process_CAMEO()
    logging.info(len(test))
    np.save(f'{DATA_ROOT}/megatron/test_dataset', test, allow_pickle=True)


# prepare_megatron()
# output: 20, 129
# exit(0)

def build_data():
    tasks = ['train', 'test']
    cmd = """/opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/{}.json \
        --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt \
        --output-prefix ./data/megatron/{} --dataset-impl mmap --workers 20"""
    # for i in tasks:
    #     os.system(cmd.format(i ,i))
    logging.info(cmd.format(tasks[0], tasks[0]))
    logging.info(cmd.format(tasks[1], tasks[1]))

# build_data()
# exit(0)
# /opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/train.json         --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt         --output-prefix ./data/megatron/train --dataset-impl mmap --workers 20
# /opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/test.json         --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt         --output-prefix ./data/megatron/test --dataset-impl mmap --workers 20


def megatron_predict():
    cmd = """CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 ./pretrain_msa.py --num-layers 12 \
        --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1 \
        --data-path ./contact/data/megatron/{}_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap \
        --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters {} --max-tokens 262144 --max-aligns 256 --max-length 1024 \
        --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save \
        --attention-name {} --finetune --attention-path ./contact/data/pred-megatron/ \
        --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release"""
    arg_dict = [['train', 21], ['test', 129]]

    # for arg in arg_dict:
    logging.info(cmd.format(arg_dict[0][0], arg_dict[0][1], arg_dict[0][0]))
    logging.info(cmd.format(arg_dict[1][0], arg_dict[1][1], arg_dict[1][0]))

# megatron_predict()
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 /dataset/ee84df8b/release/ProteinLM/pretrain/pretrain_msa.py --num-layers 12         --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1         --data-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/megatron/train_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap         --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters 21 --max-tokens 262144 --max-aligns 256 --max-length 1024         --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save         --attention-name train_256 --finetune --attention-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/pred-megatron/         --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 /dataset/ee84df8b/release/ProteinLM/pretrain/pretrain_msa.py --num-layers 12         --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1         --data-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/megatron/test_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap         --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters 129 --max-tokens 262144 --max-aligns 256 --max-length 1024         --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save         --attention-name test_256 --finetune --attention-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/pred-megatron/         --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release


def load_predictions():
    train_data = torch.load('/workspace/dump/0_train.pt')[:-13]
    # logging.info(len(train_data))
    test_data = torch.load('/workspace/dump/0_test.pt')
    logging.info(len(train_data))
    logging.info(len(test_data))

# load_predictions()
# exit(0)



# # # # # # # # # # # #
import os
import sys
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple
import torch
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
import logging
import argparse
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:\t%(message)s"
)

DATA_ROOT = './data/contact-data/'
# DATA_ROOT = './corpus/'
parser = argparse.ArgumentParser(description="Parser for sklearn solver choice, MSA depth, and checkpoint etc.")
parser.add_argument(
    "--msa-depth", type=int, help="the depth of MSA"
)
parser.add_argument(
    "--solver", type=str, choices=['liblinear', 'saga'], help="skelarn solver used for fitting logistic regression model, choose from ['liblinear', 'saga']"
)
parser.add_argument(
    "--iter", type=int, help="take the checkpoint @iter"
)
parser.add_argument(
    "--job-num", type=int, default=16, help="the number of jobs in proba prediction"
)
parser.add_argument(
    "--model-scale", type=str, choices=['1b', 'esm', '100m', '140m', '60m'], help="model scale"
)


args = parser.parse_args()
logging.info(args)
msa_depth = args.msa_depth
sklearn_solver = args.solver
ckpt_iter = args.iter
job_num = args.job_num
model_scale = args.model_scale
# if model_scale == '1b':
if model_scale == '1b' or model_scale == 'esm':
    max_len = 768
elif model_scale == '100m' or model_scale == '140m' or model_scale == '60m':
    max_len = 1024

alphabet_str = 'ARNDCQEGHILKMFPSTWYV-'
id_to_char = dict()

for i in range(len(alphabet_str)):
    id_to_char[i] = alphabet_str[i]


def map_id_to_token(token_id):
    return id_to_char[token_id]


def rename_names():
    path = "/dataset/ee84df8b/RaptorX-3DModeling/MSA_test/"
    for i in range(40):
        f = os.listdir(os.path.join(path, f"{i:04d}"))
        file = os.path.join(path, f"{i:04d}", f[0].replace("'", "\\\'"))
        os.system(f"cp {file} ./cb513/test/{i}.a2m")
    path = "/dataset/ee84df8b/RaptorX-3DModeling/MSA_valid/"
    for i in range(224):
        f = os.listdir(os.path.join(path, f"{i:04d}"))
        file = os.path.join(path, f"{i:04d}", f[0].replace("'", "\\\'"))
        os.system(f"cp {file} ./cb513/valid/{i}.a2m")

# rename_names()


def prepare_megatron(split='test'):
    def process_one_sample(idx):
        msa_path = f'./cb513/{split}'
        label_path = f'./cb513/cb513_{split}.npy'
        msa = open(os.path.join(msa_path, str(idx) + '.a2m'), 'r').read().splitlines()
        # with open(os.path.join(label_path, file_name + '.native.pkl'), 'rb') as f:
        #     labels = pickle.load(f, encoding="bytes")
        labels = np.load(label_path, allow_pickle=True)

        # dict_keys(['id', 'primary', 'contact', 'tertiary', 'valid_mask'])
        # print(msa[0].strip())
        # print(labels[idx]['primary'])
        assert msa[0].strip() == labels[idx]['primary']

        return {
            'name': labels[idx]['id'].decode(),
            'msa': msa,
            'binary_labels': labels[idx]['contact'],
        }

    if split == 'test':
        num_sample = 40
    if split == 'valid':
        num_sample = 224

    def process_CAMEO():
        trRosetta_data = []
        # label_path = f'{DATA_ROOT}/CAMEO-GroundTruth'
        # names = [i.split('.')[0] for i in os.listdir(label_path)]
        names = [i for i in range(num_sample)]
        with open(f'corpus/cb513/cb513_{split}.json', 'w') as f:
            for name in names:
                data = process_one_sample(name)
                # logging.info(data['msa'])
                one_seq = data['msa'][0] + '|' + ''.join(data['msa'][1:])
                one_seq = ' '.join(one_seq)
                f.write('{"text": ' + '"{}"}}'.format(one_seq) + "\n")

                trRosetta_data.append(data)

        return np.array(trRosetta_data, dtype=object)

    test = process_CAMEO()
    logging.info(len(test))
    np.save(f'{DATA_ROOT}/cb513_{split}_dataset', test, allow_pickle=True)

# prepare_megatron()
# prepare_megatron('valid')
# exit(0)

# prepare_megatron()
# output: 20, 129
# exit(0)

def build_data():
    tasks = ['train', 'test']
    cmd = """/opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/{}.json \
        --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt \
        --output-prefix ./data/megatron/{} --dataset-impl mmap --workers 20"""
    # for i in tasks:
    #     os.system(cmd.format(i ,i))
    logging.info(cmd.format(tasks[0], tasks[0]))
    logging.info(cmd.format(tasks[1], tasks[1]))

# build_data()
# exit(0)
# /opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/train.json         --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt         --output-prefix ./data/megatron/train --dataset-impl mmap --workers 20
# /opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/test.json         --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt         --output-prefix ./data/megatron/test --dataset-impl mmap --workers 20

# /opt/conda/bin/python ./tools/preprocess_data.py --input ./corpus/cb513/cb513_test.json         --tokenizer-type BertWordPieceCase --vocab-file ./msa_tools/msa_vocab.txt         --output-prefix ./corpus/cb513/test --dataset-impl mmap --workers 20
# /opt/conda/bin/python ./tools/preprocess_data.py --input ./corpus/cb513/cb513_valid.json         --tokenizer-type BertWordPieceCase --vocab-file ./msa_tools/msa_vocab.txt         --output-prefix ./corpus/cb513/valid --dataset-impl mmap --workers 56

def megatron_predict():
    cmd = """CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 ./pretrain_msa.py --num-layers 12 \
        --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1 \
        --data-path ./contact/data/megatron/{}_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap \
        --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters {} --max-tokens 262144 --max-aligns 256 --max-length 1024 \
        --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save \
        --attention-name {} --finetune --attention-path ./contact/data/pred-megatron/ \
        --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release"""
    arg_dict = [['train', 21], ['test', 129]]

    # for arg in arg_dict:
    logging.info(cmd.format(arg_dict[0][0], arg_dict[0][1], arg_dict[0][0]))
    logging.info(cmd.format(arg_dict[1][0], arg_dict[1][1], arg_dict[1][0]))

# megatron_predict()
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 /dataset/ee84df8b/release/ProteinLM/pretrain/pretrain_msa.py --num-layers 12         --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1         --data-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/megatron/train_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap         --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters 21 --max-tokens 262144 --max-aligns 256 --max-length 1024         --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save         --attention-name train_256 --finetune --attention-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/pred-megatron/         --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 /dataset/ee84df8b/release/ProteinLM/pretrain/pretrain_msa.py --num-layers 12         --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1         --data-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/megatron/test_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap         --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters 129 --max-tokens 262144 --max-aligns 256 --max-length 1024         --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save         --attention-name test_256 --finetune --attention-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/pred-megatron/         --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release


def load_predictions():
    train_data = torch.load('/workspace/dump/0_train.pt')[:-13]
    # logging.info(len(train_data))
    test_data = torch.load('/workspace/dump/0_test.pt')
    logging.info(len(train_data))
    logging.info(len(test_data))

# load_predictions()
# exit(0)
