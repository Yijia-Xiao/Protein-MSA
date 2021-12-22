import os

stat = dict()
# root = '/dataset/ee84df8b/MSA_20T/MSA_BL128_4/MSA_BL128_4/'
root = '/dataset/ee84df8b/MSA_20T/'
folders = os.listdir(root)

lens = []
deps = []
import tqdm
import json


def stat(fold):
    to_append = []
    for f in os.listdir(os.path.join(root, fold)):
        with open(os.path.join(root, fold, f), 'r') as fin:
            data = fin.readlines()
            lens.append(len(data[1].strip()) - 1)
            deps.append(len(data) // 2)
            to_append.append((f, lens[-1], deps[-1]))
    # json.dump(to_append, open(os.path.join(f"/dataset/ee84df8b/Protein-MSA/stat/{fold}.json"), 'w'))
    return to_append

from joblib import Parallel, delayed

job_num = 352 // 2
parallel = Parallel(n_jobs=job_num, batch_size=1)
data = parallel(delayed(stat)(fold) for fold in folders)

