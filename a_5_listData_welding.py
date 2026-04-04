import numpy as np
import random
import os

data_path='./IMG_DATA_LS//DataforSVAE/IMG'
#data_path='./IMG_DATA_LS/IMG_DATA_PEACH_0912/POSE_TXT'
temp = os.listdir(data_path)
print(temp)
np.savetxt('./IMG_DATA_LS/DataforSVAE/all.txt', temp, fmt='%s')




