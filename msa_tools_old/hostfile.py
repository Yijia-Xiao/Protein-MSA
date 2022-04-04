import os
import json

if os.path.exists("/home/hostfile.json"):
    with open("/home/hostfile.json") as file:
        hosts = json.load(file)
    for idx, host in enumerate(hosts):
        print(f'{host["ip"]}\tnode{idx}')

