# import biopython

from Bio import SeqIO
from joblib import Parallel, delayed
from tqdm import tqdm
import pathlib
import subprocess
from pathlib import Path
import os


def process(idx: int, id: str, seq: str, task: str, split: str):
    # if len(seq) > 1024:
    #     return
    folder = f"/dataset/ee84df8b/Protein-MSA/finetune/output/MSA/{task}/{split}"
    # Path(folder).mkdir(exist_ok=True)
    os.system(f'mkdir -p {folder}')

    if pathlib.Path(f"{folder}/{id}.a2m").exists():
        return
    with open(f"{folder}/{id}.fasta","w") as f:
        f.write("> id\n" + seq)

    cmd = f"/bin/bash BuildFeatures/HHblitsWrapper/BuildMSA4DistPred.sh -n 3 -c 1 -o {folder} {folder}/{id}.fasta"
    subprocess.run(cmd.split())
    # # BuildFeatures/A3MToA2M.sh -o ./ sample.a3m
    cmd = f"/bin/bash BuildFeatures/A3MToA2M.sh -o {folder} {folder}/{id}.a3m"
    subprocess.run(cmd.split())

    pathlib.Path(f"{folder}/{id}.fasta").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.fasta_raw").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.a3m").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.seq").unlink(missing_ok=True)

# for i in range(1000):
#     pathlib.Path("MSA/%04d" % i).mkdir(parents=True, exist_ok=True)

def build(task, split):
    print(f"/dataset/ee84df8b/Protein-MSA/finetune/output/{task}/{task}_{split}.fasta")
    Parallel(n_jobs=58)(delayed(process)(idx, r.id, str(r.seq), task, split) for idx, r in tqdm(enumerate(SeqIO.parse(f"/dataset/ee84df8b/Protein-MSA/finetune/output/{task}/{task}_{split}.fasta", "fasta"))))

import sys
task = int(sys.argv[1])
print(task)
# exit(0)

if task == 0:
    tasks = ['proteinnet']
    splits = ['train', 'valid', 'test']
if task == 1:
    tasks = ['secondary_structure']
    splits = ['train', 'valid', 'casp12', 'ts115', 'cb513']
if task == 2:
    tasks = ['remote_homology']
    splits = ['train', 'valid', 'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout']
if task == 3:
    tasks = ['fluorescence']
    splits = ['train']
if task == 4:
    tasks = ['fluorescence']
    splits = ['valid']
if task == 5:
    tasks = ['fluorescence']
    splits = ['test']
if task == 6:
    tasks = ['stability']
    splits = ['test']
if task == 7:
    tasks = ['stability']
    splits = ['valid']
if task == 8:
    tasks = ['stability']
    splits = ['train']

# tasks = ['fluorescence', 'stability', 'proteinnet']
# splits = ['train', 'valid', 'test']
# 'secondary_structure'
# ['train', 'valid', 'casp12', 'ts115', 'cb513']
# 'remote_homology'
# ['train', 'valid', 'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout']

print(task, tasks, splits)

for t in tasks:
    for s in splits:
        build(t, s)



# (base) root@task6c44af49169ab24c9f7edb86698780c2-v2-train-master-59pjn:/dataset/ee84df8b/Protein-MSA/finetune/output# ls fluorescence/*.fasta | xargs wc -l
#    54434 fluorescence/fluorescence_test.fasta
#    42892 fluorescence/fluorescence_train.fasta
#    10724 fluorescence/fluorescence_valid.fasta
#   108050 total
# (base) root@task6c44af49169ab24c9f7edb86698780c2-v2-train-master-59pjn:/dataset/ee84df8b/Protein-MSA/finetune/output# ls stability/*.fasta | xargs wc -l
#   25702 stability/stability_test.fasta
#  107228 stability/stability_train.fasta
#    5024 stability/stability_valid.fasta
#  137954 total
# (base) root@task6c44af49169ab24c9f7edb86698780c2-v2-train-master-59pjn:/dataset/ee84df8b/Protein-MSA/finetune/output# ls remote_homology/*.fasta | xargs wc -l
#    2544 remote_homology/remote_homology_test_family_holdout.fasta
#    1436 remote_homology/remote_homology_test_fold_holdout.fasta
#    2508 remote_homology/remote_homology_test_superfamily_holdout.fasta
#   24624 remote_homology/remote_homology_train.fasta
#    1472 remote_homology/remote_homology_valid.fasta
#   32584 total
# (base) root@task6c44af49169ab24c9f7edb86698780c2-v2-train-master-59pjn:/dataset/ee84df8b/Protein-MSA/finetune/output# ls secondary_structure/*.fasta | xargs wc -l
#      42 secondary_structure/secondary_structure_casp12.fasta
#    1026 secondary_structure/secondary_structure_cb513.fasta
#   17356 secondary_structure/secondary_structure_train.fasta
#     230 secondary_structure/secondary_structure_ts115.fasta
#    4340 secondary_structure/secondary_structure_valid.fasta
#   22994 total
# (base) root@task6c44af49169ab24c9f7edb86698780c2-v2-train-master-59pjn:/dataset/ee84df8b/Protein-MSA/finetune/output# ls proteinnet/*.fasta | xargs wc -l
#      80 proteinnet/proteinnet_test.fasta
#   50598 proteinnet/proteinnet_train.fasta
#     448 proteinnet/proteinnet_valid.fasta
#   51126 total




# tasks = ['fluorescence', 'stability', 'proteinnet']
# splits = ['train', 'valid', 'test']
