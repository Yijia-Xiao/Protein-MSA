import argparse
import math
from tqdm import tqdm
from Bio.SeqIO.FastaIO import Seq, SeqRecord
import lmdb

from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random

import json
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        # env = lmdb.open(str(data_file), max_readers=1, readonly=True,
        #                 lock=False, readahead=False, meminit=False)

        env = lmdb.open(str(data_file), max_readers=1, readonly=False,
                        lock=False, readahead=False, meminit=False, map_size=int(1e9))

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    # def get_msa():
    #     id_fill = math.ceil(math.log10(len(dataset)))
    #     for i, element in enumerate(tqdm(dataset)):
    #         id_ = element.get('id', str(i).zfill(id_fill))

    #         if isinstance(id_, bytes):
    #             id_ = id_.decode()
    #         # print(id_)
    #         global cnt
    #         cnt += 1
    #         if cnt <= 2:
    #             print(element)
    #         primary = element['primary']
    #         seq = Seq(primary)
    #         with open(f'{msa_path}/{str(i).zfill(id_fill)}.a2m', 'r') as f:
    #             msa = [li.strip() for li in f.readlines()]
    #         assert msa[0].strip() == seq, 'mismatch in query sequence'


    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        # if self._in_memory and self._cache[index] is not None:
        #     item = self._cache[index]
        # else:
        # with self._env.begin(write=False) as txn:
        with self._env.begin(write=True) as txn:
            item = pkl.loads(txn.get(str(index).encode()))
            if 'id' not in item:
                item['id'] = str(index)
            print('item', item)
            # txn.put(str(index).encode(), 'hello'.encode())
            if self._in_memory:
                self._cache[index] = item
        return item



def calc(item):
    # protein_length = len(item['primary'])
    token_ids = item['primary']
    input_mask = np.ones_like(token_ids)
    valid_mask = item['valid_mask']
    contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)

    yind, xind = np.indices(contact_map.shape)
    invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
    invalid_mask |= np.abs(yind - xind) < 6
    contact_map[invalid_mask] = -1

    return contact_map, squareform(pdist(item['tertiary']))


cnt = 0
def process(task, split):
    # Path(f'./output/{task}').mkdir(exist_ok=True)
    lmdbfile = f'./data-msa/{task}/{task}_{split}.lmdb'
    msa_path = f'./output/MSA/{task}/{split}'
    dataset = LMDBDataset(lmdbfile)

    data = list()
    id_fill = math.ceil(math.log10(len(dataset)))
    for i, element in enumerate(tqdm(dataset)):
        id_ = element.get('id', str(i).zfill(id_fill))

        if isinstance(id_, bytes):
            id_ = id_.decode()
        # print(id_)
        global cnt
        cnt += 1
        if cnt <= 2:
            print(element)
        primary = element['primary']
        seq = Seq(primary)
        with open(f'{msa_path}/{str(i).zfill(id_fill)}.a2m', 'r') as f:
            msa = [li.strip() for li in f.readlines()]
        assert msa[0].strip() == seq, 'mismatch in query sequence'
        # print(msa, seq)
        # element['msa'] = msa
        data.append(element)
        # print(dir(element))
        # print(element)
        exit(0)

    # with dataset._env.begin(write=True) as txn:
    #     for i, element in enumerate(tqdm(dataset)):
    #         id_ = element.get('id', str(i).zfill(id_fill))

    #         if isinstance(id_, bytes):
    #             id_ = id_.decode()
    #         # print(id_)
    #         primary = element['primary']
    #         seq = Seq(primary)
    #         with open(f'{msa_path}/{str(i).zfill(id_fill)}.a2m', 'r') as f:
    #             msa = [li.strip() for li in f.readlines()]
    #         assert msa[0].strip() == seq, 'mismatch in query sequence'
    #         # print(msa, seq)
    #         element['msa'] = msa
    #         data.append(element)
    #         # print(dir(element))
    #     txn.put(element.get('id', str(i).zfill(id_fill)), msa)

    return data


tasks = ['fluorescence', 'stability', 'proteinnet']
splits = ['train', 'valid', 'test']
for t in tasks:
    for s in splits:
        data = process(t, s)

for s in ['train', 'valid', 'casp12', 'ts115', 'cb513']:
    data = process('secondary_structure', s)

for s in ['train', 'valid', 'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout']:
    data = process('remote_homology', s)
