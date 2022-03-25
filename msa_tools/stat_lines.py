from joblib import Parallel, delayed
import json
import os
import subprocess


root = "/dataset/ee84df8b/data/JSON/"

def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 2 ** 31
    # buffer = 1024 ** 3
    # buffer = 1024 * 1024
    with open(os.path.join(root, file_name)) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        print([file_name, sum(buf.count('\n') for buf in buf_gen)])
        return [file_name, sum(buf.count('\n') for buf in buf_gen)]



def split_512():
    def split_json(idx):
        file_name = f"/dataset/ee84df8b/data/JSON/MSA_AB384BL512_{idx}.json"
        # cmd = f"split -l 125000 {os.path.join(root, file_name)} -d -a 1 /dataset/ee84df8b/data/BIN/MSA_AB384BL512_{idx}_"
        cmd = f"split -l 125000 {os.path.join(root, file_name)} -d -a 1 /dataset/ee84df8b/data/SPLIT/MSA_AB384BL512_{idx}_"
        print(cmd)
        subprocess.run(cmd.split())
    job_num = 8
    parallel = Parallel(n_jobs=job_num, batch_size=1)
    stat = parallel(delayed(split_json)(i) for i in range(1, 9))

split_512()

# MSA_AB384BL512_0.json    
split = """MSA_AB128BL256_0.json  MSA_AB128BL256_4.json  MSA_AB256BL384_0.json  MSA_AB256BL384_4.json  MSA_AB256BL384_8.json    MSA_AB384BL512_3.json  MSA_AB384BL512_7.json  MSA_BL128_2.json
MSA_AB128BL256_1.json  MSA_AB128BL256_5.json  MSA_AB256BL384_1.json  MSA_AB256BL384_5.json  MSA_AB384BL512_4.json  MSA_AB384BL512_8.json  MSA_BL128_3.json
MSA_AB128BL256_2.json  MSA_AB128BL256_6.json  MSA_AB256BL384_2.json  MSA_AB256BL384_6.json  MSA_AB384BL512_1.json    MSA_AB384BL512_5.json  MSA_BL128_0.json       MSA_BL128_4.json
MSA_AB128BL256_3.json  MSA_AB128BL256_7.json  MSA_AB256BL384_3.json  MSA_AB256BL384_7.json  MSA_AB384BL512_2.json    MSA_AB384BL512_6.json  MSA_BL128_1.json"""


# files = split.split()
# job_num = 31
# parallel = Parallel(n_jobs=job_num, batch_size=1)

# for s in files:
#     stat = parallel(delayed(iter_count)(f) for f in files)

# print(stat)
# json.dump(stat, open('file_lines.json', 'w'))
