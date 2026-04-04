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
from PoseTransfer import RawDataTransfer


def splitDate(data_path='./IMG_DATA', rate=0.8):
    """分割数据"""
    abs_path = os.path.join(data_path,"IMG_OUTSIDE")
    temp = os.listdir(abs_path)
    train_num = int(len(temp)*rate)
    train_sample = random.sample(temp,train_num)
    var_sample = list(set(temp)-set(train_sample))
    np.savetxt(os.path.join(data_path,"train.txt"), train_sample, fmt='%s')
    np.savetxt(os.path.join(data_path, "var.txt"), var_sample, fmt='%s')


class ReconstDataset(Dataset):
    def __init__(self, data_path="./IMG_DATA/train.txt", scaled_width = 320, scale_height= 320):
        super().__init__()
        self.data = []

        temp_img_list = np.loadtxt(data_path, dtype=str)
        # absolute path, check file exits
        path,filename = os.path.split(data_path)
        for j in range(len(temp_img_list)):
            temp = os.path.join(path,"IMG_OUTSIDE",temp_img_list[j])
            temp1 = temp.replace("IMG_OUTSIDE", "IMG_LEFT")
            temp2 = temp.replace("IMG_OUTSIDE", "IMG_RIGHT")
            temp3 = temp.replace("IMG_OUTSIDE", "POSE_TXT")
            temp3 = temp3.replace('jpg','txt')
            if os.path.isfile(temp) == False or os.path.isfile(temp1) == False or os.path.isfile(temp2) == False or os.path.isfile(temp3) == False:
                continue
            print("loop:",j,temp,os.path.isfile(temp),os.path.isfile(temp1),os.path.isfile(temp2),os.path.isfile(temp3))
            # check data format
            temp4 = np.loadtxt(temp3)
            if len(temp4.shape)==1:
                continue

            self.data.append(temp)


        self.data = np.array(self.data)
        self.data = self.data.reshape(-1)
        self.width = scaled_width
        self.height = scale_height
        self.PoseReader = RawDataTransfer(init_path=path+"/Pose_Init")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path_abs = self.data[item]
        # # 外部相机
        # img_data = cv2.imread(img_path_abs)
        # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        # img_data = cv2.resize(img_data, dsize=(self.width, self.height))
        # 左相机
        temp1 = img_path_abs.replace("IMG_OUTSIDE", "IMG_LEFT")
        img1_data = cv2.imread(temp1)
        img1_data = cv2.cvtColor(img1_data, cv2.COLOR_BGR2GRAY)
        # img1_data = cv2.normalize(img1_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img1_data = img1_data/255.0
        img1_data = cv2.resize(img1_data, dsize=(self.width, self.height))

        # 右相机
        temp2 = img_path_abs.replace("IMG_OUTSIDE", "IMG_RIGHT")
        img2_data = cv2.imread(temp2)
        img2_data = cv2.cvtColor(img2_data, cv2.COLOR_BGR2GRAY)
        # img2_data = cv2.normalize(img2_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img2_data = img2_data/255.0
        img2_data = cv2.resize(img2_data, dsize=(self.width, self.height))

        # pose label
        temp3 = img_path_abs.replace("IMG_OUTSIDE", "POSE_TXT")
        temp3 = temp3.replace('jpg','txt')
        joint_config, objectPose = self.PoseReader.run(temp3)

        print("img1_tensor shape:", torch.Tensor([img1_data]).shape)  
        print("img2_tensor shape:", torch.Tensor([img2_data]).shape)    
        #print("img3_tensor shape:", torch.Tensor([img3_data]).shape)        
        print("joint_config shape:", torch.Tensor([joint_config]).shape)  
        print("objectPose shape:", torch.Tensor([objectPose]).shape) 

        # label_data = {'image': img_data}
        # return torch.Tensor([img1_data])
        # return torch.from_numpy(img1_data), torch.from_numpy(img2_data), torch.from_numpy(joint_config), torch.from_numpy(objectPose)
        return torch.Tensor(np.array([img1_data])), torch.Tensor(np.array([img2_data])), torch.Tensor(np.array([joint_config])), torch.Tensor(np.array([objectPose]))


class ClassificationDataset(Dataset):
    def __init__(self, data_path="./IMG_DATA/", data_mode = "train",scaled_width = 320, scale_height= 320):
        super().__init__()
        self.data = []
        folder_list = glob.glob(data_path+"/*")
        if data_mode == "train":
            data_file = "train.txt"
        else:
            data_file = "var.txt"

        for i in range(len(folder_list)):
            subfolder = folder_list[i]+"/"+data_file

            temp_img_list = np.loadtxt(subfolder, dtype=str)
            # absolute path, check file exits
            # path,filename = os.path.split(subfolder)
            for j in range(len(temp_img_list)):
                temp = os.path.join(folder_list[i],"IMG_OUTSIDE",temp_img_list[j])
                temp1 = temp.replace("IMG_OUTSIDE", "IMG_LEFT")
                temp2 = temp.replace("IMG_OUTSIDE", "IMG_RIGHT")
                temp3 = temp.replace("IMG_OUTSIDE", "POSE_TXT")
                temp3 = temp3.replace('jpg','txt')
                if os.path.isfile(temp) == False or os.path.isfile(temp1) == False or os.path.isfile(temp2) == False or os.path.isfile(temp3) == False:
                    continue
                # check data format
                temp4 = np.loadtxt(temp3)
                if len(temp4.shape)==1:
                    continue

                self.data.append(temp)


        self.data = np.array(self.data)
        self.data = self.data.reshape(-1)
        self.width = scaled_width
        self.height = scale_height
        self.PoseReader = RawDataTransfer(init_path=folder_list[i]+"/Pose_Init")

    def defineCls(self, file_name:str):
        if "circle" in file_name:
            cls = 0
        elif "quadra" in file_name:
            cls = 1
        elif "tri" in file_name:
            cls = 2
        elif "tube1" in file_name:
            cls = 3
        elif "tube2" in file_name:
            cls = 4
        elif "tube3" in file_name:
            cls = 5
        elif "tube4" in file_name:
            cls = 6        
        return cls        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path_abs = self.data[item]
        # # 外部相机
        # img_data = cv2.imread(img_path_abs)
        # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        # img_data = cv2.resize(img_data, dsize=(self.width, self.height))
        # 左相机
        temp1 = img_path_abs.replace("IMG_OUTSIDE", "IMG_LEFT")
        img1_data = cv2.imread(temp1)
        img1_data = cv2.cvtColor(img1_data, cv2.COLOR_BGR2GRAY)
        # img1_data = cv2.normalize(img1_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img1_data = img1_data/255.0
        img1_data = cv2.resize(img1_data, dsize=(self.width, self.height))

        # 右相机
        temp2 = img_path_abs.replace("IMG_OUTSIDE", "IMG_RIGHT")
        img2_data = cv2.imread(temp2)
        img2_data = cv2.cvtColor(img2_data, cv2.COLOR_BGR2GRAY)
        # img2_data = cv2.normalize(img2_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img2_data = img2_data/255.0
        img2_data = cv2.resize(img2_data, dsize=(self.width, self.height))

        # class label

        temp_label = self.defineCls(img_path_abs)
        temp_label = np.array([temp_label])
        
        # pose label
        temp3 = img_path_abs.replace("IMG_OUTSIDE", "POSE_TXT")
        temp3 = temp3.replace('jpg','txt')
        joint_config, _ = self.PoseReader.run(temp3)

        # label_data = {'image': img_data}
        # return torch.Tensor([img1_data])
        # return torch.from_numpy(img1_data), torch.from_numpy(img2_data), torch.from_numpy(joint_config), torch.from_numpy(objectPose)
        return torch.Tensor(np.array([img1_data])), torch.Tensor(np.array([img2_data])), torch.Tensor(np.array([joint_config])), torch.from_numpy(temp_label)




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




