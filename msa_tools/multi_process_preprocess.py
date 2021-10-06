import os
import sys
import tqdm
import multiprocessing
from multiprocessing import Manager, Process, Pool

# like "UniRef50-xa-a2m-2017"
# UniRef50-xa-a2m-2017
# UniRef50-xb-a2m-2018
# UniRef50-xc-a2m-2017
# UniRef50-xd-a2m-2018
# UniRef50-xe-a2m-2017
# UniRef50-xf-a2m-2018

split_name = sys.argv[1]
print(f'start {split_name}')
data_path = f'/dataset/ee84df8b/20210816/protein/data/data/{split_name}'

out_folder = f'/workspace/data/{split_name}'
os.system(f'mkdir -p {out_folder}')
out_file = f'{out_folder}/{split_name}.json'

files = os.listdir(data_path)

NUM_THREAD = 52

MAX_DEPTH = 512
MAX_LENGTH = 1024

def process_file(proc_idx, file_list):
    for fd in tqdm.tqdm(file_list):
        sample = ""
        with open(os.path.join(data_path, fd)) as f:
            msa = f.readlines()[:MAX_DEPTH]
            split_add = False
            for align in msa:
                sample += align.strip()[:MAX_LENGTH]
                if not split_add:
                    sample += '|'
                    split_add = True
        p_str = ' '.join(sample)
        with open(os.path.join(out_folder, str(proc_idx)), 'a') as fout:
            fout.write('{"text": "' + p_str + ' "}\n')


num_sample_per_split = len(files) // NUM_THREAD
pool = multiprocessing.Pool(processes=NUM_THREAD)

for worker_idx in range(NUM_THREAD):
    # ret = pool.apply_async(read_func, (worker_idx, data_path, file_list_split[worker_idx], depth, length))
    # ret_lis.append(ret)
    pool.apply_async(process_file, (worker_idx, files[worker_idx * num_sample_per_split:
                                                      (worker_idx + 1) * num_sample_per_split]))

pool.close()
pool.join()
cat_cmd = '/bin/cat ' + ' '.join([os.path.join(out_folder, str(idx)) for idx in range(NUM_THREAD)]) + '> ' + out_file
# print(cat_cmd)
os.system(cat_cmd)

# manager = Manager()
# return_dict = manager.dict()

# jobs = []
#
# for worker_idx in range(NUM_THREAD):
#     return_dict[worker_idx] = list()
#     p = multiprocessing.Process(target=process_file,
#                                 args=(worker_idx,
#                                       files[worker_idx * num_sample_per_split: (worker_idx + 1) * num_sample_per_split],
#                                       return_dict))
#     jobs.append(p)
#     p.start()
#
# for proc in jobs:
#     proc.join()
#
# for worker_idx in range(NUM_THREAD):
#     for s in tqdm.tqdm(return_dict[worker_idx]):
#         p_str = ' '.join(s)
#         print('{"text": "' + p_str + ' "}')

