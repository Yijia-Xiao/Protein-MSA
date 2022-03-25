import json
import os
import numpy as np
import matplotlib.pyplot as plt  


l_1 = []
l_2 = []
l_5 = []
max_iter = 180000
# for i in range(53):
#     with open(f'./log/corp-{100000 + 500 * i}-dp128-lib-job32.log', 'r') as fin:
max_steps = max_iter // 500

for i in range(112 + 66):
    with open(f'./log/corp-{70500 + 500 * i}-dp128-lib-job32.log', 'r') as fin:
        data = [l for l in fin.readlines()][-1]
        p = data[25:].split()
        hits = []
        ratio = []
        for idx in [2, 9, 16]:
            hits.append(int(p[idx][:-1]))
            hits.append(int(p[idx + 2][:-1]))
            ratio.append(round(hits[-2] / hits[-1], 4))
        l_1.append(ratio[-3])
        l_2.append(ratio[-2])
        l_5.append(ratio[-1])

def plot(data, legend):
    x = np.arange(70500, 159500, 500)
    plt.plot(x, data, 'ro-')
    plt.title(legend)
    plt.xlabel('Iter')
    plt.ylabel('Precision')
    plt.savefig(legend)


plot(l_1, legend='l1')
plot(l_2, legend='l2')
plot(l_5, legend='l5')

def find_max(arr):
    print(np.argmax(np.array(arr)))
    print(np.max(np.array(arr)))

for i in [l_1, l_2, l_5]:
    find_max(i)


