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
    # cmd = f"/bin/bash BuildFeatures/A3MToA2M.sh -o {folder} {folder}/{id}.a3m"
    subprocess.run(cmd.split())
    pathlib.Path(f"{folder}/{id}.fasta").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.fasta_raw").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.a3m").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.seq").unlink(missing_ok=True)

# for i in range(1000):
#     pathlib.Path("MSA/%04d" % i).mkdir(parents=True, exist_ok=True)

def build(task, split):
    print(f"/dataset/ee84df8b/Protein-MSA/finetune/output/{task}/{task}_{split}.fasta")
    Parallel(n_jobs=48)(delayed(process)(idx, r.id, str(r.seq), task, split) for idx, r in tqdm(enumerate(SeqIO.parse(f"/dataset/ee84df8b/Protein-MSA/finetune/output/{task}/{task}_{split}.fasta", "fasta"))))

tasks = ['fluorescence', 'stability', 'proteinnet']
splits = ['train', 'valid', 'test']

for t in tasks:
    for s in splits:
        build(t, s)
for s in ['train', 'valid', 'casp12', 'ts115', 'cb513']:
    build('secondary_structure', s)
for s in ['train', 'valid', 'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout']:
    build('remote_homology', s)


