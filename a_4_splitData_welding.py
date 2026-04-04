import numpy as np
import random
import os

def splitDate(data_path='./IMG_DATA_LS/DataforSVAE/IMG', rate=0.8):
    temp = os.listdir(data_path)
    train_num = int(len(temp)*rate)
    train_sample = random.sample(temp,train_num)
    var_sample = list(set(temp)-set(train_sample))
    np.savetxt('./IMG_DATA_LS/DataforSVAE/train.txt', train_sample, fmt='%s')
    np.savetxt('./IMG_DATA_LS/DataforSVAE/var.txt', var_sample, fmt='%s')


splitDate("./IMG_DATA_LS/DataforSVAE/IMG")
