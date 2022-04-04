import os
import sys
import tqdm
import multiprocessing
from multiprocessing import Manager, Process, Pool
import json
from joblib import Parallel, delayed
import subprocess
import math
import sys

data_path = '../proteinnet/valid/'
out_path = '../proteinnet/valid'
bin_path = '../proteinnet/valid'

NUM_THREAD = 32
MAX_DEPTH = 2048
parallel = Parallel(n_jobs=NUM_THREAD, batch_size=1)

def process_sample(data_path, fd):
    with open(os.path.join(data_path, fd)) as f:
        msa = f.readlines()[:MAX_DEPTH]
        sample = ""
        try:
            sample = msa[0].strip() + '|'
        except:
            print(data_path, fd)
        for align in msa[1:]:
            sample += align.strip()
        p_str = ' '.join(sample)
    return json.dumps({'text': p_str})


def process_file(data_path):
    files = os.listdir(data_path)
    # print(files)
    # data = parallel(delayed(process_file)(data_path, fd) for fd in files)
    # print(data)
    with open(out_path + '.json', 'a') as f:
        for fd in files:
            sample = process_sample(data_path, fd)
            f.write(sample + '\n')

    cmd = f"""/opt/conda/bin/python ../tools/preprocess_data.py --input {out_path + '.json'} \
            --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
            --output-prefix {bin_path} --dataset-impl mmap --workers {NUM_THREAD}"""
    print(cmd)
    subprocess.run(cmd.split())

# /opt/conda/bin/python ../tools/preprocess_data.py --input ../proteinnet/valid.json --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt --output-prefix ../proteinnet/valid --dataset-impl mmap --workers 32

process_file(data_path)
