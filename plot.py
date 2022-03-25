#!/usr/bin/env python
# coding: utf-8


import torch
import matplotlib.pyplot as plt



def data_distribution(path='./data/contact-data/CAMEO-GroundTruth/'):
    import os
    import pickle
    import matplotlib.pyplot as plt

    files = os.listdir(path)
    length_list = []
    for f in files:
        with open(os.path.join(path, f), 'rb') as f:
            item = pickle.load(f, encoding="bytes")
            # print(item.keys())
            # print(item['sequence'.encode()].decode())
            length_list.append(len(item['sequence'.encode()].decode()))
    # print(files)

    print(length_list)
    print(sum(length_list) / len(length_list))

    plt.hist(length_list, bins = 20)
    plt.savefig('cameo-stat.png')
    plt.close()


def a2m_distribution(path='./data/contact-data/CAMEO-trRosettaA2M'):
    import os
    import matplotlib.pyplot as plt

    files = os.listdir(path)
    length_list = []
    aligns_list = []
    for f in files:
        with open(os.path.join(path, f), 'r') as f:
            data = f.readlines()
            aligns_list.append(len(data))
            length_list.append(len(data[0].strip()))

    length_list = sorted(length_list)
    aligns_list = sorted(aligns_list)
    # print(length_list)
    # print(aligns_list)
    print(sum(length_list) / len(length_list), length_list[len(length_list) // 2])
    print(sum(aligns_list) / len(aligns_list), aligns_list[len(aligns_list) // 2])

    plt.figure()
    plt.hist(length_list, bins = 20)
    plt.savefig('cameo-len.png')
    plt.close()

    plt.figure()
    plt.hist(aligns_list, bins = 20)
    plt.savefig('cameo-dep.png')
    plt.close()
 

# data_distribution()
a2m_distribution()
exit(0)

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

# def plot_iter_model(iters, model, depth=128, long=False):
#     megatron_iters = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/{model}-fp32-depth{depth}-{iters}-train.pt')
#     plt.figure(figsize=(12, 9))
#     plt.title(f'Contact from iter {iters}, model {model}, depth {depth}', fontsize=28, color='w')
#     if not long:
#         if model == '1b':
#             plt.imshow(apc(symmetrize(megatron_iters[147][8][1:, 1:].float().softmax(dim=-1))), cmap='Blues')
#         elif model == '140m':
#             plt.imshow(apc(symmetrize(megatron_iters[89][0][1:, 1:].float().softmax(dim=-1))), cmap='Blues')
#         elif model == '60m':
#             plt.imshow(apc(symmetrize(megatron_iters[69][7][1:, 1:].float().softmax(dim=-1))), cmap='Blues')
#     else:
#         if model == '1b':
#             plt.imshow(apc(symmetrize(megatron_iters[147 - 15 * 7][8][1:, 1:].float().softmax(dim=-1))), cmap='Blues')
#         elif model == '140m':
#             plt.imshow(apc(symmetrize(megatron_iters[89 - 9 * 7][0][1:, 1:].float().softmax(dim=-1))), cmap='Blues')
#         elif model == '60m':
#             plt.imshow(apc(symmetrize(megatron_iters[69 - 7 * 7][7][1:, 1:].float().softmax(dim=-1))), cmap='Blues')
#     plt.xlabel(f'Contact-iter-{iters}-model-{model}-depth-{depth}-long-{long}', fontsize=20)
#     plt.savefig(f'./cmap/Contact-iter-{iters}-model-{model}-depth-{depth}-long-{long}.png', bbox_inches='tight')


# plot_iter_model(80000, '1b')
# plot_iter_model(14000, '140m')
# plot_iter_model(10000, '60m')
# plot_iter_model(10000, '60m', depth=256)

# plot_iter_model(80000, '1b', long=True)
# plot_iter_model(14000, '140m', long=True)
# plot_iter_model(10000, '60m', long=True)
# plot_iter_model(10000, '60m', depth=256, long=True)


# plot_iter_model(92500, '1b', long=True)
# plot_iter_model(22500, '140m', long=True)
# plot_iter_model(24500, '60m', long=True)

# plot_iter_model(24500, '60m', long=True)

# plot_iter_model(14000, '140m')

# plot_iter_model(97500, '1b', long=True)

# plot_iter_model(10000, '60m')


# for idx, i in enumerate(megatron_iters[1::9]):
#     print(len(i[0]), idx)

# len(megatron_iters[81][0]), len(megatron_iters[82][0])

# for i in range(8):
#     plt.figure()
#     plt.imshow(apc(symmetrize(megatron_iters[89][i][1:, 1:].float().softmax(dim=-1))), cmap='Blues')


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def vis_1b(idx, split='cameo', iters=100500):
    # ret = torch.load('/dataset/ee84df8b/Protein-MSA/data/attention/ret/1b-128-97500.pt')
    if split == 'cameo':
        ret = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/ret/cameo-1b-fp32-depth128-{iters}-train.pt')
    elif split == 'casp12test':
        ret = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/ret/casp12-1b-fp32-depth128-{iters}-test.pt')
    elif split == 'casp12valid':
        ret = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/ret/casp12-1b-fp32-depth128-{iters}-vaild.pt')

    plt.figure(figsize=(12, 12))
    plt.title(f"Contact Map from 1B-Model", fontsize=24)
    ret[idx][1][ret[idx][1] < 0] = 0
    for i in range(ret[idx][1].size(0)):
        for j in range(ret[idx][1].size(0)):
            if ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 1 and                 ret[idx][1][i][j] == 1:
                s1 = plt.scatter(i, j, color='green') # , facecolor='none')
            elif ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 1 and                 ret[idx][1][i][j] == 0:
                s2 = plt.scatter(i, j, color='red', facecolor='none', marker="*", alpha=0.5)
            elif ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 0 and                 ret[idx][1][i][j] == 1:
                s3 = plt.scatter(i, j, color='blue', facecolor='none', marker="*", alpha=0.5)
    plt.legend((s1, s2, s3), ('Hit', 'False Positive', 'False Negative'), loc="best")
    plt.savefig(f'./imgs/1b/{split}-{iters}-{idx}.png')

def vis_esm(idx, split='cameo'):
    if split == 'cameo':
        ret = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/1b-fp32-depth128-{iters}-train.pt')
    elif split == 'casp12test':
        ret = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/cb-test-1b-fp32-depth128-{iters}-train.pt')
        ret = torch.load('./data/attention/cb-test-esm-fp32-depth128-10-valid.pt')
    elif split == 'casp12valid':
        ret = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/cb-valid-1b-fp32-depth128-{iters}-train.pt')

    plt.figure(figsize=(12, 12))
    plt.title(f"ESM Contact Map", fontsize=24)
    ret[idx][1][ret[idx][1] < 0] = 0
    for i in range(ret[idx][1].size(0)):
        for j in range(ret[idx][1].size(0)):
            if ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 1 and                 ret[idx][1][i][j] == 1:
                s1 = plt.scatter(i, j, color='green') # , facecolor='none')
            elif ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 1 and                 ret[idx][1][i][j] == 0:
                s2 = plt.scatter(i, j, color='red', facecolor='none', marker="*", alpha=0.5)
            elif ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 0 and                 ret[idx][1][i][j] == 1:
                s3 = plt.scatter(i, j, color='blue', facecolor='none', marker="*", alpha=0.5)
    plt.legend((s1, s2, s3), ('Hit', 'False Positive', 'False Negative'), loc="best")
    plt.savefig(f'esm-{idx}.png')


vis_1b(3), vis_esm(3)

vis_1b(27), vis_esm(27)

exit(0)

def vis(idx):
    plt.figure(figsize=(12, 12))
    plt.title(f"Contact Map")
    ret[idx][1][ret[idx][1] < 0] = 0
    for i in range(ret[idx][1].size(0)):
        for j in range(ret[idx][1].size(0)):
            if ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 1 and                 ret[idx][1][i][j] == 1:
                s1 = plt.scatter(i, j, color='green') # , facecolor='none')
            elif ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 1 and                 ret[idx][1][i][j] == 0:
                s2 = plt.scatter(i, j, color='red', facecolor='none', marker="*", alpha=0.5)
            elif ret[idx][0].reshape(ret[idx][1].shape)[i][j] == 0 and                 ret[idx][1][i][j] == 1:
                s3 = plt.scatter(i, j, color='blue', facecolor='none', marker="*", alpha=0.5)
    plt.legend((s1, s2, s3), ('Hit', 'False Positive', 'False Negative'),                loc="best")
    plt.show()
vis(27)


# In[155]:





# In[156]:


vis(3)


# In[157]:


vis(27)


# In[140]:


def find_ok(length):
    for idx, r in enumerate(ret):
        if r[1].size(0) == length:
            print(idx)
            return idx
# for l in [378, 345, 354, 242, 211, 165, 140, 242]:
for l in [334, 405]:
    test(find_ok(l))


# In[106]:


test(3)


# In[95]:


test(32)


# In[96]:


test(104)


# In[92]:


test(123)


# In[93]:


test(23)


# In[82]:


test(123)


# In[83]:


test(31)


# In[84]:


test(89)


# In[86]:


# for i in ret:
#     print(i[1].size(0))


# In[58]:


# set(ret[0][1].reshape(-1).tolist())


# In[56]:


test(26)


# In[24]:


plt.imshow(ret[0][1].reshape((382, 382)), cmap='Blues')


# In[ ]:


ret[0]


# In[4]:


plot_1b(13500)


# In[5]:


plot_1b(16500)


# In[6]:


plot_1b(20000)


# In[7]:


plot_1b(25500)


# In[8]:


plot_1b(31500)


# In[9]:


plot_1b(36500)


# In[10]:


plot_1b(40500)


# In[11]:


plot_1b(44500)


# In[12]:


plot_1b(45000)


# In[13]:


plot_1b(50000)


# In[14]:


plot_1b(53500)


# In[22]:


plot_1b(54500)


# In[23]:


plot_1b(69000)


# In[53]:


plot_1b(80000)


# In[ ]:





# In[51]:


plot_140m(7000)


# In[ ]:





# In[ ]:


iters = 36500
megatron_iters = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/1b-fp32-depth128-{iters}-train.pt')
plt.figure(figsize=(12, 9))
plt.title(f'Contact from iter {iters}, 1b model, depth 128', fontsize=28, color='w')
plt.imshow(apc(symmetrize(megatron_iters[147][8][1:, 1:].float().softmax(dim=-1))), cmap='Blues')


# In[9]:


megatron_iters = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/1b-fp32-depth128-25500-train.pt')
plt.figure(figsize=(12, 9))
plt.title(f'Contact from iter 25500, 1b model, depth 128', fontsize=28, color='w')
plt.imshow(apc(symmetrize(megatron_iters[147 + 15][8][1:, 1:].float().softmax(dim=-1))), cmap='Blues')


# In[19]:


megatron_iters = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/1b-fp32-depth128-25500-train.pt')
plt.figure(figsize=(12, 9))
plt.title(f'Contact from iter 25500, 1b model, depth 128', fontsize=28, color='w')
plt.imshow(apc(symmetrize(megatron_iters[147 - 15 * 7][8][1:, 1:].float().softmax(dim=-1))), cmap='Blues')


# In[32]:


megatron_iters = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/1b-fp32-depth128-31500-train.pt')
plt.figure(figsize=(12, 9))
plt.title(f'Contact from iter 31500, 1b model, depth 128', fontsize=28, color='w')
plt.imshow(apc(symmetrize(megatron_iters[147 - 15 * 7][8][1:, 1:].float().softmax(dim=-1))), cmap='Blues')


# In[16]:


def plot_large(iter_):
    megatron_iters = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/1b-fp32-depth128-{iter_}-train.pt')
    plt.figure(figsize=(12, 9))
    plt.title(f'Contact from iter {iter_}, 1b model, depth 128', fontsize=28, color='w')
    plt.imshow(apc(symmetrize(megatron_iters[147 - 15 * 7][8][1:, 1:].float().softmax(dim=-1))), cmap='Blues')


# In[17]:


plot_large(36500)


# In[18]:


plot_large(40500)


# In[19]:


plot_large(44500)


# In[20]:


plot_large(53500)


# In[21]:


plot_large(54500)


# In[24]:


plot_large(69000)


# In[54]:


plot_large(80000)


# In[30]:


plt.figure(figsize=(12, 9))
plt.title(f'Contact from iter 25500, 1b model, depth 128', fontsize=28, color='w')
plt.imshow(apc(symmetrize(megatron_iters[147 - 15 * 2][8][1:, 1:].float().softmax(dim=-1))), cmap='Blues')


# In[20]:


len(megatron_iters[147 - 15 * 7][8][1:, 1:])


# In[ ]:





# In[28]:


megatron_100m = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/megatron_89500_train_depth128.pt')
plt.figure(figsize=(12, 9))
plt.title(f'Contact from 100M 89500', fontsize=28, color='w')
plt.imshow(apc(symmetrize(megatron_100m[38][-6][1:, 1:].float().softmax(dim=-1))), cmap='Blues')


# In[40]:


plot_large(40500)


# In[26]:


for i in megatron_100m[:: 13]:
    print(len(i[0][0]))


# In[ ]:


# megatron_48000 = torch.load('/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/megatron_48000_train_depth128.pt')
# plt.figure(figsize=(12, 9))
# plt.title('Contact from iter 48k', fontsize=28, color='w')
# plt.imshow(apc(symmetrize(megatron_48000[129][-6].softmax(dim=-1))), cmap='Blues')

