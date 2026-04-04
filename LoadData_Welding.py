# import imp
import os
# from cv2 import waitKey
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import PIL
from PIL import Image, ImageOps
import cv2
# import skimage
import glob
import numpy as np
import random
from PoseTransfer_Nochange import RawDataTransfer


def splitDate(data_path='./IMG', rate=0.8):
    """分割数据"""
    abs_path = os.path.join(data_path,"IMG")
    temp = os.listdir(abs_path)
    train_num = int(len(temp)*rate)
    train_sample = random.sample(temp,train_num)
    var_sample = list(set(temp)-set(train_sample))
    np.savetxt(os.path.join(data_path,"train.txt"), train_sample, fmt='%s')
    np.savetxt(os.path.join(data_path, "var.txt"), var_sample, fmt='%s')


class ReconstDataset(Dataset):
    def __init__(self, data_path="./IMG/train.txt", scaled_width = 320, scale_height= 320):
        super().__init__()
        self.data = []
        #print("ReconstDataset ====================================:")
        temp_img_list = np.loadtxt(data_path, dtype=str)
        # absolute path, check file exits
        path,filename = os.path.split(data_path)
        for j in range(len(temp_img_list)):
            temp = os.path.join(path,"IMG",temp_img_list[j])
            temp1 = temp.replace("IMG", "IMG")
            temp4 = temp.replace("IMG", "POSE_TXT").replace('jpg', 'txt')

            #print("loop:",j,temp,os.path.isfile(temp),os.path.isfile(temp1),os.path.isfile(temp2),os.path.isfile(temp3),os.path.isfile(temp4))

            if os.path.isfile(temp) == False or os.path.isfile(temp1) == False or os.path.isfile(temp4) == False:
                continue
            # check data format
            temp4 = np.loadtxt(temp4)
            
            if len(temp4.shape)==1:
                continue

            self.data.append(temp)
        #print("ReconstDataset ====================================:")

        self.data = np.array(self.data)
        #print("ReconstDataset ====================================:",len(self.data))
        self.data = self.data.reshape(-1)
        ##print("ReconstDataset ====================================:")
        self.width = scaled_width
        self.height = scale_height
        #print("ReconstDataset ====================================:")
        #self.PoseReader = RawDataTransfer(init_path=path+"/Pose_Init")
        #print("PoseReader = RawDataTransfer(init_path=path====================================:")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        #print("__getitem__ ====================================:")
        img_path_abs = self.data[item]
        # # 外部相机
        # img_data = cv2.imread(img_path_abs)
        # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        # img_data = cv2.resize(img_data, dsize=(self.width, self.height))
        # 左相机
        temp1 = img_path_abs.replace("IMG", "IMG")
        img1_data = cv2.imread(temp1)
        img1_data = cv2.cvtColor(img1_data, cv2.COLOR_BGR2GRAY)
        # img1_data = cv2.normalize(img1_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img1_data = img1_data/255.0
        img1_data = cv2.resize(img1_data, dsize=(self.width, self.height))

        # pose label
        temp4 = img_path_abs.replace("IMG", "POSE_TXT")
        temp4 = temp4.replace('jpg','txt')
        objectPose = temp4


        return torch.Tensor(np.array([img1_data])),  torch.Tensor(np.array([objectPose]))



def listFun(batch):
    data_list = []
    for i in range(len(batch)):
        _img = batch[i]['image']
        data_list.append([_img])
    return torch.Tensor(data_list)



if __name__ == "__main__":
    #拆分数据集
    # splitDate('./IMG_DATA_tri')
    # exit()

    ww = ReconstDataset(data_path="./IMG_DATA/IMG_DATA_tube1/train.txt")
    m = DataLoader(dataset=ww, batch_size=8, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
    print(len(ww))
    preMax = torch.Tensor([[-10e5,-10e5,-10e5,-10e5,-10e5,-10e5]])
    preMin = torch.Tensor([[10e5,10e5,10e5,10e5,10e5,10e5]])
    for i_batch, data  in enumerate(m):
        imgLeft= data[0]
        imgRight= data[1]
        joint_config = data[2]
        objectPose = data[3]
        # 24 torch.Size([2, 1, 360, 640]) torch.Size([2, 1, 360, 640]) torch.Size([2, 1]) torch.Size([2, 1, 6])
        # print(i_batch, imgLeft.shape, imgRight.shape, joint_config.shape,objectPose.shape)
        # print(torch.cat((imgLeft, imgRight),axis = 0).shape)
        # print(imgLeft[0][0]/255)
        # cv2.imshow("a", (imgLeft[0][0]/255).cpu().detach().numpy())
        # cv2.waitKey()
        
        temp_max = torch.max(objectPose, dim=0)[0]
        temp_min = torch.min(objectPose, dim=0)[0]
        preMax = torch.cat((preMax, temp_max),axis=0)
        preMin = torch.cat((preMin, temp_min),axis=0)
        

    print(torch.max(preMax, dim=0)[0], torch.min(preMin, dim=0)[0])
    print(torch.max(preMax, dim=0)[0] - torch.min(preMin, dim=0)[0])


