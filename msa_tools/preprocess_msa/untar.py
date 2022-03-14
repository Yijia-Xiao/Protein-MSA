import tarfile
import os

def un_tar(file_path):
    file_name = file_path.strip().split('/')[-1][:-4]
    tar = tarfile.open(file_path)
    # os.system(f'mkdir -p {file_name}')
    tar.extractall(path=file_name)
    tar.close()

filestr = """/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_6.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_1.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_8.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_0.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_4.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_3.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_2.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_5.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_3.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_4.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_5.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_2.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_8.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_1.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_6.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_7.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_0.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_1.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_6.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_7.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_0.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_3.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_4.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_5.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_2.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/MSA_AB384BL512_7.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_3.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_4.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_2.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_1.tar
/dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_0.tar"""


from joblib import Parallel, delayed
files = filestr.split('\n')

job_num = 31
parallel = Parallel(n_jobs=job_num, batch_size=1)
data = parallel(delayed(un_tar)(fold) for fold in files)



# files = """/dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_6.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_1.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_8.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_0.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_4.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_3.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_2.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB384BL512/MSA_AB384BL512_5.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_3.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_4.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_5.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_2.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_8.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_1.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_6.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_7.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA/AB256BL384/MSA_AB256BL384_0.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_1.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_6.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_7.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_0.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_3.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_4.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_5.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/AB128BL256/MSA_AB128BL256_2.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/MSA_AB384BL512_7.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_3.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_4.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_2.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_1.tar
# /dataset/ee84df8b/MSA_30T/MSA/MSA_2/BL128/MSA_BL128_0.tar"""


# for i in files.split('\n'):
#     # print(i)
#     fname = i.split('/')[-1][:-4]
#     # tar xf MSA_BL128_4.tar  -C MSA_BL128_4
#     print(f"mkdir -p {fname}\ntar xf {i} -C {fname}")
