# import biopython

from Bio import SeqIO
from joblib import Parallel, delayed
from tqdm import tqdm
import pathlib
import subprocess

def process(idx: int, id: str, seq: str):
    if len(seq) > 1024:
        return
    folder = "MSA/%04d" % (idx % 1000)
    if pathlib.Path(f"{folder}/{id}.a2m").exists():
        return
    with open(f"{folder}/{id}.fasta","w") as f:
        f.write("> id\n" + seq)
        # -o {id} 
    cmd = f"/bin/bash BuildFeatures/HHblitsWrapper/BuildMSA4DistPred.sh -n 3 -c 1 -o {folder} {folder}/{id}.fasta"
    subprocess.run(cmd.split())
    # BuildFeatures/A3MToA2M.sh -o ./ sample.a3m
    cmd = f"/bin/bash BuildFeatures/A3MToA2M.sh -o {folder} {folder}/{id}.a3m"
    subprocess.run(cmd.split())
    pathlib.Path(f"{folder}/{id}.fasta").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.fasta_raw").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.a3m").unlink(missing_ok=True)
    pathlib.Path(f"{folder}/{id}.seq").unlink(missing_ok=True)

for i in range(1000):
    pathlib.Path("MSA/%04d" % i).mkdir(parents=True, exist_ok=True)

Parallel(n_jobs=68)(delayed(process)(idx, r.id, str(r.seq)) for idx, r in tqdm(enumerate(SeqIO.parse("/root/uniref90.fasta", "fasta"))))

