import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


with open('stat.json', 'r') as f:
    data = json.load(f)
aligns = []
length = []
for split, file_list in data.items():
    # print(len(file_list))
    for f in file_list.values():
        aligns.append(f[0])
        length.append(f[1])

import statistics
print(sum(aligns) / len(aligns))
print(statistics.median(aligns))
print(sum(length) / len(length))
print(statistics.median(length))
# print(len(aligns))

# items = [6, 1, 8, 2, 3]

# statistics.median(items)
# exit(0)


def plot(data, labl):
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    x = data
    # plt.hist(x, density=True, bins=100, label=f"MSA {labl}")
    plt.hist(x, bins=100, label=f"MSA {labl}")
    # mn, mx = plt.xlim()
    # plt.xlim(mn, mx)
    # kde_xs = np.linspace(mn, mx, 300)
    # kde = st.gaussian_kde(x)
    # plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    plt.legend(loc="upper right")
    if labl == 'Aligns':
        plt.xlabel("MSA Depth", fontsize=16)
    else:
        plt.xlabel("MSA Sequence Length", fontsize=16)
    plt.ylabel("Number of Samples", fontsize=16)
    if labl == 'Aligns':
        plt.title("MSA Depth Histogram", fontsize=16)
    else:
        plt.title("MSA Length Histogram", fontsize=16)
    plt.savefig(f'{labl}', format='pdf')
    plt.clf()

plot([i for i in aligns if i < 6000], 'Aligns')
plot([i for i in length if i < 4000], 'Length')

print(f'{max(aligns)=}')
print(f'{max(length)=}')

exit(0)

def analyze_best_example():
    import json
    with open('eval_dict_list.json', 'r') as f:
        data = json.load(f)
    # print(data[0])
    # print(len(data))
    for sample in data:
        # if sample['long']['1']['tot'] > 320 and sample['long']['5']['cor'] / sample['long']['5']['tot'] > 0.95:
        # if sample['long']['1']['tot'] > 120 and sample['long']['5']['cor'] / sample['long']['5']['tot'] > 0.8 and sample['long']['5']['cor'] / sample['long']['5']['tot'] < 0.9:
        if sample['long']['1']['tot'] > 200 and sample['long']['1']['cor'] / sample['long']['1']['tot'] > 0.85:
            print(sample)

analyze_best_example()
# {'short': {'1': {'cor': 98, 'tot': 378}, '2': {'cor': 70, 'tot': 189}, '5': {'cor': 46, 'tot': 75}}, 'mid': {'1': {'cor': 196, 'tot': 378}, '2': {'cor': 145, 'tot': 189}, '5': {'cor': 71, 'tot': 75}}, 'long': {'1': {'cor': 278, 'tot': 378}, '2': {'cor': 170, 'tot': 189}, '5': {'cor': 75, 'tot': 75}}, 'midlong': {'1': {'cor': 318, 'tot': 378}, '2': {'cor': 181, 'tot': 189}, '5': {'cor': 75, 'tot': 75}}, 'all': {'1': {'cor': 378, 'tot': 378}, '2': {'cor': 189, 'tot': 189}, '5': {'cor': 75, 'tot': 75}}}
# {'short': {'1': {'cor': 176, 'tot': 345}, '2': {'cor': 130, 'tot': 172}, '5': {'cor': 69, 'tot': 69}}, 'mid': {'1': {'cor': 199, 'tot': 345}, '2': {'cor': 130, 'tot': 172}, '5': {'cor': 66, 'tot': 69}}, 'long': {'1': {'cor': 293, 'tot': 345}, '2': {'cor': 164, 'tot': 172}, '5': {'cor': 69, 'tot': 69}}, 'midlong': {'1': {'cor': 309, 'tot': 345}, '2': {'cor': 168, 'tot': 172}, '5': {'cor': 69, 'tot': 69}}, 'all': {'1': {'cor': 345, 'tot': 345}, '2': {'cor': 172, 'tot': 172}, '5': {'cor': 69, 'tot': 69}}}
# {'short': {'1': {'cor': 132, 'tot': 354}, '2': {'cor': 88, 'tot': 177}, '5': {'cor': 46, 'tot': 70}}, 'mid': {'1': {'cor': 234, 'tot': 354}, '2': {'cor': 144, 'tot': 177}, '5': {'cor': 58, 'tot': 70}}, 'long': {'1': {'cor': 266, 'tot': 354}, '2': {'cor': 165, 'tot': 177}, '5': {'cor': 70, 'tot': 70}}, 'midlong': {'1': {'cor': 300, 'tot': 354}, '2': {'cor': 161, 'tot': 177}, '5': {'cor': 70, 'tot': 70}}, 'all': {'1': {'cor': 354, 'tot': 354}, '2': {'cor': 177, 'tot': 177}, '5': {'cor': 70, 'tot': 70}}}


# 3
# {'short': {'1': {'cor': 80, 'tot': 165}, '2': {'cor': 52, 'tot': 82}, '5': {'cor': 24, 'tot': 33}}, 'mid': {'1': {'cor': 122, 'tot': 165}, '2': {'cor': 64, 'tot': 82}, '5': {'cor': 30, 'tot': 33}}, 'long': {'1': {'cor': 98, 'tot': 165}, '2': {'cor': 66, 'tot': 82}, '5': {'cor': 33, 'tot': 33}}, 'midlong': {'1': {'cor': 129, 'tot': 165}, '2': {'cor': 78, 'tot': 82}, '5': {'cor': 33, 'tot': 33}}, 'all': {'1': {'cor': 165, 'tot': 165}, '2': {'cor': 82, 'tot': 82}, '5': {'cor': 33, 'tot': 33}}}

# 27
# {'short': {'1': {'cor': 124, 'tot': 334}, '2': {'cor': 99, 'tot': 167}, '5': {'cor': 56, 'tot': 66}}, 'mid': {'1': {'cor': 158, 'tot': 334}, '2': {'cor': 110, 'tot': 167}, '5': {'cor': 54, 'tot': 66}}, 'long': {'1': {'cor': 286, 'tot': 334}, '2': {'cor': 163, 'tot': 167}, '5': {'cor': 64, 'tot': 66}}, 'midlong': {'1': {'cor': 300, 'tot': 334}, '2': {'cor': 163, 'tot': 167}, '5': {'cor': 64, 'tot': 66}}, 'all': {'1': {'cor': 334, 'tot': 334}, '2': {'cor': 167, 'tot': 167}, '5': {'cor': 66, 'tot': 66}}}
