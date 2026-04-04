import numpy as np
import random
import os

def splitDate(data_path='./IMG_DATA_LS/IMG_DATA_J2_250801/IMG_BEHIND', rate=0.8):
    temp = os.listdir(data_path)
    train_num = int(len(temp)*rate)
    train_sample = random.sample(temp,train_num)
    var_sample = list(set(temp)-set(train_sample))
    np.savetxt('./IMG_DATA_LS/IMG_DATA_J2_250801/train.txt', train_sample, fmt='%s')
    np.savetxt('./IMG_DATA_LS/IMG_DATA_J2_250801/var.txt', var_sample, fmt='%s')


splitDate("./IMG_DATA_LS/IMG_DATA_J2_250801/IMG_BEHIND")
