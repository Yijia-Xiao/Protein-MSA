import json
import subprocess
from joblib import Parallel, delayed
import tqdm
import pathlib
import os
import random
import shutil

def add_sub():
    collect = json.load(open('/dataset/ee84df8b/Protein-MSA/stat/collect.json', 'r'))
    new = dict()
    for spl, v in collect.items():
        a2m_spl = spl.replace('MSA/', 'A2M/')
        new[a2m_spl] = dict()
        for sub in v:
            name = sub[0][0]
            idx = name.split('_')[-1].split('.')[0][:-4]
            new[a2m_spl][idx] = [[i[0][:-3] + 'a2m', i[1], i[2]] for i in sub]
    json.dump(new, open('/dataset/ee84df8b/Protein-MSA/stat/a2m_with_path.json', 'w'))

# add_sub()
# exit(0)

prefix = "/dataset/ee84df8b/data/A2M/"
def merge_and_generate_list():
    collect = json.load(open('/dataset/ee84df8b/Protein-MSA/stat/a2m_with_path.json', 'r'))
    file_lists = list()
    for spl, sub_list in collect.items():
        for sub, a2m_name_len_dep in sub_list.items():
            file_lists += [(os.path.join(spl, sub, triple[0]), triple[0], triple[1], triple[2]) for triple in a2m_name_len_dep]
    json.dump(file_lists, open('/dataset/ee84df8b/Protein-MSA/stat/A2M_file_lists.json', 'w'))

# merge_and_generate_list()
# exit(0)

def shuffle_list():
    collect = json.load(open('/dataset/ee84df8b/Protein-MSA/stat/A2M_file_lists.json', 'r'))
    random.shuffle(collect)
    json.dump(collect, open('/dataset/ee84df8b/Protein-MSA/stat/A2M_shuffle_lists.json', 'w'))
    print(len(collect))

# shuffle_list()
# exit(0)


file_list = json.load(open('/dataset/ee84df8b/Protein-MSA/stat/A2M_shuffle_lists.json', 'r'))

num_files = len(file_list)
num_splits = 100
per_split = num_files // num_splits

def split_list():
    ret = []
    for i in range(0, num_files, per_split):
        ret.append(file_list[i: i + per_split])
    ret = ret[:-1]
    print([len(l) for l in ret])
    return ret

file_splits = split_list()
from pathlib import Path

def copy_split(idx, file_split):
    target_path = f'/dataset/ee84df8b/data/SHUFFLE/{idx:02d}'
    # os.mkdir(target_path)
    Path(target_path).mkdir(exist_ok=True)
    # target_path.mkdir(exist_ok=True)
    for src_path, name, l, d in tqdm.tqdm(file_split):
        tgt_path = Path(os.path.join(target_path, name))
        if tgt_path.exists():
            continue
        try:
            shutil.copy(src_path, os.path.join(target_path, name))
        except:
            with open('log', 'a') as f:
                f.write(src_path + target_path + '\n')


parallel = Parallel(n_jobs=50)

parallel(delayed(copy_split)(idx, file_split) for idx, file_split in enumerate(file_splits))

# import shutil
# def copy(src, dst):
#     pass

# def convert(spl, sub, name):
#     cmd = f"bash BuildFeatures/A3MToA2M.sh -o /dataset/ee84df8b/data/A2M/{spl}/{sub} /dataset/ee84df8b/data/MSA/{spl}/{sub}/{name}"
#     subprocess.run(cmd.split())

# parallel = Parallel(n_jobs=64)

# for spl, v in collect.items():
#     spl = spl.split('/')[-1]
#     for sub, lis in tqdm(v.items()):
#         parallel(delayed(convert)(spl, sub, name[0]) for name in lis)

