import tarfile
import os

filestr = """/dataset/ee84df8b/data/MSA/MSA_AB384BL512_0
/dataset/ee84df8b/data/MSA/MSA_AB384BL512_1
/dataset/ee84df8b/data/MSA/MSA_AB384BL512_2
/dataset/ee84df8b/data/MSA/MSA_AB384BL512_3
/dataset/ee84df8b/data/MSA/MSA_AB384BL512_4
/dataset/ee84df8b/data/MSA/MSA_AB384BL512_5
/dataset/ee84df8b/data/MSA/MSA_AB384BL512_6
/dataset/ee84df8b/data/MSA/MSA_AB384BL512_7
/dataset/ee84df8b/data/MSA/MSA_AB384BL512_8
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_0
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_1
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_2
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_3
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_4
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_5
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_6
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_7
/dataset/ee84df8b/data/MSA/MSA_AB256BL384_8
/dataset/ee84df8b/data/MSA/MSA_AB128BL256_0
/dataset/ee84df8b/data/MSA/MSA_AB128BL256_1
/dataset/ee84df8b/data/MSA/MSA_AB128BL256_2
/dataset/ee84df8b/data/MSA/MSA_AB128BL256_3
/dataset/ee84df8b/data/MSA/MSA_AB128BL256_4
/dataset/ee84df8b/data/MSA/MSA_AB128BL256_5
/dataset/ee84df8b/data/MSA/MSA_AB128BL256_6
/dataset/ee84df8b/data/MSA/MSA_AB128BL256_7
/dataset/ee84df8b/data/MSA/MSA_BL128_0
/dataset/ee84df8b/data/MSA/MSA_BL128_1
/dataset/ee84df8b/data/MSA/MSA_BL128_2
/dataset/ee84df8b/data/MSA/MSA_BL128_3
/dataset/ee84df8b/data/MSA/MSA_BL128_4"""



import tqdm
import json
import os
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

files = filestr.split('\n')

collect = json.load(open('stat/collect.json', 'r'))

orpha = 0
tot = 0
for k, v in collect.items():
    for s in v:
        for i in s:
            if i[2] == 1:
                orpha += 1
            tot += 1
    print(orpha)
print(orpha, tot)


def run_stat():

    job_num = 48
    def stat_split(root):
        stat = dict()
        folders = os.listdir(root) # [:8]
        def stat(fold):
            to_append = []
            for f in os.listdir(os.path.join(root, fold)):
                with open(os.path.join(root, fold, f), 'r') as fin:
                    data = fin.readlines()
                    # lens.append(len(data[1].strip()) - 1)
                    # deps.append(len(data) // 2)
                    to_append.append((f, len(data[1].strip()) - 1, len(data) // 2))
            return to_append
        parallel = Parallel(n_jobs=job_num, batch_size=1)
        data = parallel(delayed(stat)(fold) for fold in folders)
        return data


    collect = dict()

    for f_split in files:
        ret = stat_split(f_split)
        json.dump(ret, open(f'/dataset/ee84df8b/Protein-MSA/stat/{f_split.split("/")[-1]}.json', 'w'))
        collect[f_split] = ret

    json.dump(collect, open(f'/dataset/ee84df8b/Protein-MSA/stat/collect.json', 'w'))



def analyze():
    collect = json.load(open(f'/dataset/ee84df8b/Protein-MSA/stat/collect.json', 'r'))
    lens = []
    deps = []

    for f_split in files:
        sub_split = collect[f_split]
        for spl in sub_split:
            for sample in spl:
                # lens.append(sample[1] + 1)
                # if sample[2] < 1024:
                if sample[2] < 2048:
                    deps.append(sample[2])

    # print(f'length min={np.min(lens)}, median={np.median(lens)}, max={np.max(lens)}, mean = {np.mean(lens)}')
    print(f'depth min={np.min(deps)}, median={np.median(deps)}, max={np.max(deps)}, mean = {np.mean(deps)}')
    print(len(deps))

    def plot(data, name):
        n, bins, patches = plt.hist(x=data, bins=100, color='#0504aa',alpha=0.7, rwidth=0.85)
        plt.xlabel('Value',fontsize=15)
        plt.ylabel('Frequency',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel('Frequency',fontsize=15)
        plt.title(name,fontsize=15)
        plt.savefig(f"{name}.png")

    # plot(lens, "MSA-Length")
    # plot(deps, "MSA-Aligns")
    plot(deps, "MSA-Aligns-Crop")
    print(len(lens))


# analyze()



# def stat(fold):
#     to_append = []
#     for subfold in os.listdir(fold):
#         for f in os.listdir(os.path.join(fold, subfold))
#             with open(os.listdir(os.path.join(fold, subfold, f)), 'r') as fin:
#                 data = fin.readlines()
#                 lens.append(len(data[1].strip()) - 1)
#                 deps.append(len(data) // 2)
#                 to_append.append((f, lens[-1], deps[-1]))
#     return to_append



# for f in files:
#     print(f'mv {f}/{f.split("/")[-1]}/* {f}/ && rmdir {f}/{f.split("/")[-1]}/')

# job_num = 31
# parallel = Parallel(n_jobs=job_num, batch_size=1)
# data = parallel(delayed(un_tar)(fold) for fold in files)
