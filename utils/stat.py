import os
import json

root = "/dataset/ee84df8b/20210816/protein/data/data/"
sub = "UniRef50-xa-a2m-2017/ UniRef50-xb-a2m-2018/ UniRef50-xc-a2m-2017/ UniRef50-xd-a2m-2018/ UniRef50-xe-a2m-2017/ UniRef50-xf-a2m-2018".split('/')

# print(sub)
stat = dict()
for i in sub:
    stat[i.strip()] = dict()
    folder_path = os.path.join(root, i.strip())
    for f_name in os.listdir(folder_path):
        # print(f_name)
        # UPI000D1436B8.a2m
        with open(os.path.join(folder_path, f_name), 'r') as f:
            data = f.readlines()
            stat[i.strip()][f_name] = (len(data), len(data[0].strip()))
        # print(stat)

json.dump(stat, open('stat.json', 'w'))
