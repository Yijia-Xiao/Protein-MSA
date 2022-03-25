import lmdb
import numpy as np
import pickle as pkl
from scipy.spatial.distance import pdist, squareform


def contact(tertiary, valid_mask):
    contact_map = np.less(squareform(pdist(tertiary)), 8.0).astype(np.int64)
    yind, xind = np.indices(contact_map.shape)
    invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
    # invalid_mask |= np.abs(yind - xind) < 6
    contact_map[invalid_mask] = -1
    return contact_map


def read(split_):
    # idx = 0
    data = list()
    env = lmdb.open(f"proteinnet/proteinnet_{split_}.lmdb", readonly=True, max_readers=16, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        for _ in range(num_examples):
            item = pkl.loads(txn.get(str(_).encode()))
            # dict_keys(['id', 'primary', 'evolutionary', 'secondary', 'tertiary', 'protein_length', 'valid_mask'])
            data.append(
                {
                    'id': item['id'],
                    'primary': item['primary'],
                    'contact': contact(item['tertiary'], item['valid_mask']),
                    'tertiary': item['tertiary'],
                    'valid_mask': item['valid_mask'],
                }
            )
    env.close()
    np.save(f'cb513_{split_}.npy', data)

    with open(f'./fasta/{split_}.fasta', 'w') as f:
        for idx, item in enumerate(data):
            f.write(f">{idx}_{item['id'].decode()}\n{item['primary']}\n")
    print(len(data))

read('train')
read('valid')
read('test')

# print(len(data))



