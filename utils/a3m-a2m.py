import json
import subprocess
from joblib import Parallel, delayed
from tqdm import tqdm
import pathlib


def add_sub():
    collect = json.load(open('/dataset/ee84df8b/Protein-MSA/stat/old-collect.json', 'r'))
    new = dict()
    for spl, v in collect.items():
        new[spl] = dict()
        for sub in v:
            name = sub[0][0]
            idx = name.split('_')[-1].split('.')[0][:-4]
            # print(name, idx)
            # break
            new[spl][idx] = sub
    json.dump(new, open('/dataset/ee84df8b/Protein-MSA/stat/collect.json', 'w'))

# add_sub()

# exit(0)

collect = json.load(open('/dataset/ee84df8b/Protein-MSA/stat/collect.json', 'r'))
def convert(spl, sub, name):
    cmd = f"bash BuildFeatures/A3MToA2M.sh -o /dataset/ee84df8b/data/A2M/{spl}/{sub} /dataset/ee84df8b/data/MSA/{spl}/{sub}/{name}"
    subprocess.run(cmd.split())


# for spl, v in collect.items():
#     spl = spl.split('/')[-1]
#     for sub in v:
#         pathlib.Path(f"/dataset/ee84df8b/data/A2M/{spl}/{sub}").mkdir(parents=True, exist_ok=True)


parallel = Parallel(n_jobs=64)

for spl, v in collect.items():
    spl = spl.split('/')[-1]
    for sub, lis in tqdm(v.items()):
        # pathlib.Path(f"/dataset/ee84df8b/data/A2M/{spl}/{sub}").mkdir(parents=True, exist_ok=True)
        parallel(delayed(convert)(spl, sub, name[0]) for name in lis)

# for k, v in collect.items():
#     for s in v:
#         for i in s:
#             if i[2] == 1:
#                 orpha += 1
#             tot += 1
#     print(orpha)
# print(orpha, tot)
