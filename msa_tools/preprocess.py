import os
import sys
import tqdm
import multiprocessing
from multiprocessing import Manager, Process, Pool
import json
from joblib import Parallel, delayed
import subprocess

root_path = '/dataset/ee84df8b/data/A2M/'
out_path = '/dataset/ee84df8b/data/JSON/'
bin_path = '/dataset/ee84df8b/data/BIN/'

splits = [
    # 'MSA_BL128_4',
    'MSA_AB384BL512_0', 'MSA_AB384BL512_1', 'MSA_AB384BL512_2', 'MSA_AB384BL512_3', 'MSA_AB384BL512_4', 'MSA_AB384BL512_5', 'MSA_AB384BL512_6', 'MSA_AB384BL512_7', 'MSA_AB384BL512_8', 
    'MSA_AB256BL384_0', 'MSA_AB256BL384_1', 'MSA_AB256BL384_2', 'MSA_AB256BL384_3', 'MSA_AB256BL384_4', 'MSA_AB256BL384_5', 'MSA_AB256BL384_6', 'MSA_AB256BL384_7', 'MSA_AB256BL384_8', 
    'MSA_AB128BL256_0', 'MSA_AB128BL256_1', 'MSA_AB128BL256_2', 'MSA_AB128BL256_3', 'MSA_AB128BL256_4', 'MSA_AB128BL256_5', 'MSA_AB128BL256_6', 'MSA_AB128BL256_7', 
    'MSA_BL128_0', 'MSA_BL128_1', 'MSA_BL128_2', 'MSA_BL128_3', 
]
NUM_THREAD = 52
MAX_DEPTH = 2048

def process_sub(split, sub_idx):
    sub_path = os.path.join(root_path, split, sub_idx)
    files = os.listdir(sub_path)
    ret = []
    for fd in tqdm.tqdm(files):
        with open(os.path.join(sub_path, fd)) as f:
            msa = f.readlines()[:MAX_DEPTH]
            sample = msa[0].strip() + '|'
            for align in msa[1:]:
                sample += align.strip()
            p_str = ' '.join(sample)
            ret.append(json.dumps({'text': p_str}))
    return ret

job_num = 48
parallel = Parallel(n_jobs=job_num, batch_size=1)

# out_file = f'{out_folder}/{split_name}.json'
# files = os.listdir(data_path)

def process_split(split):
    print(f'start process {split}')
    cmd = f"""/opt/conda/bin/python ../tools/preprocess_data.py --input {out_path + split + '.json'} \
            --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
            --output-prefix {bin_path + split} --dataset-impl mmap --workers {job_num}"""
    print(cmd)
    split_path = os.path.join(root_path, split)
    subs = os.listdir(split_path)
    data = parallel(delayed(process_sub)(split, sub) for sub in subs)
    to_dump = []
    for d in data:
        to_dump += d
    print(f"len for {split} = {len(to_dump)}")
    with open(out_path + split + '.json', 'w') as f:
        # json.dump(to_dump, f)
        for l in to_dump:
            f.write(l + '\n')
    subprocess.run(cmd.split())
    

for spl in splits:
    process_split(spl)
