import json
import os
import numpy as np
import matplotlib.pyplot as plt  


l_1 = []
l_2 = []
l_5 = []
start_iter = 70500
# max_iter = # 180000
max_iter = 217000
max_iter = 250000
# max_steps = max_iter // 500
max_steps = (max_iter - start_iter) // 500


for i in range(max_steps):
    with open(f'./log/corp-{70500 + 500 * i}-dp128-lib-job32.log', 'r') as fin:
        data = [l for l in fin.readlines()][-1]
        p = data[25:].split()
        hits = []
        ratio = []
        for idx in [2, 9, 16]:
            hits.append(int(p[idx][:-1]))
            hits.append(int(p[idx + 2][:-1]))
            ratio.append(round(hits[-2] / hits[-1], 4))
        # print(p[2], p[4], p[9], p[11], p[16], p[18])
        # print(json.loads(p))
        # print(hits)
        l_1.append(ratio[-3])
        l_2.append(ratio[-2])
        l_5.append(ratio[-1])
    # print(ratio)


# print(l_1)
def plot(data, legend):
    # x = np.arange(70500, 126500, 500)
    x = np.arange(70500, max_iter, 500)
    # l1=plt.plot(x1,y1,'r--',label='type1')
    # l2=plt.plot(x2,y2,'g--',label='type2')
    # l3=plt.plot(x3,y3,'b--',label='type3')
    plt.plot(x, data, 'ro-')
    plt.title(legend)
    plt.xlabel('Iter')
    plt.ylabel('Precision')
    # plt.legend()
    # plt.show()
    plt.savefig(legend)


plot(l_1, legend='l1')
plot(l_2, legend='l2')
plot(l_5, legend='l5')


def find_max(arr):
    print(np.argmax(np.array(arr)))
    print(np.max(np.array(arr)))
    return np.argmax(np.array(arr))


max_arg_set = set()

for i in [l_1, l_2, l_5]:
    max_arg_set.add(find_max(i))

for ag in max_arg_set:
    print(ag)
    for i in [l_1, l_2, l_5]:
        print(i[ag], end=', ')
    print('')



# files = os.listdir('./log')
# for f in files:
#     if f[:4] == 'corp':
#         with open(f'./log/corp-{100000 + 500 * i}-dp128-lib-job32.log' + f, 'r') as fin:
#             data = [l for l in fin.readlines()][-1]
#             p = data[25:]
#             print(p)
#             # print(json.loads(p))
