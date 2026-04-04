import os
import numpy as np
data_path="./IMG_DATA_cycle/train.txt"
temp_img_list = np.loadtxt(data_path, dtype=str)
# print(temp_img_list)

#
path,filename = os.path.split(data_path)
for j in range(len(temp_img_list)):
    temp = os.path.join(path,"IMG_OUTSIDE",temp_img_list[j])
    temp1 = temp.replace("IMG_OUTSIDE", "IMG_LEFT")
    temp2 = temp.replace("IMG_OUTSIDE", "IMG_RIGHT")
    if os.path.isfile(temp) == False:
        print(temp)

    if os.path.isfile(temp1) == False:
        print(temp1)

    if os.path.isfile(temp2) == False:
        print(temp2)

