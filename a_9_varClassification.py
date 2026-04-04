"""
@Modified: 
@Description: 固定encoder, 训练回归网络
"""

import torch
import torch.nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import math
import yaml
from ClassificationNet import CNet, EncordFun
from LoadData import ClassificationDataset

from scipy.spatial.transform import Rotation as spR
import cv2
import copy


class TrainClassification(object):
    def __init__(self):
        self.batch_size = 512
        self.learning_rate = 0.001
        self.momentum = 0.3
        self.epoch = 100
        self.pre_model_pth = "./weights/reconstruction_weight_LS_64.pth"
        self.saved_pth = "./weights/pose_weight_LS.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        # loss
        self.recons_loss =torch.nn.MSELoss()
        self.data_folder = "./IMG_DATA_LS"
        self.width = 320
        self.height = 320

    @torch.no_grad()
    def var(self, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ClassificationDataset(data_path=self.data_folder, data_mode = "var", scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        var_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        reload_model = CNet(self.pre_model_pth).to(self.device)
        # if self.device.type != 'cpu':
        #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
        reload_model.load_state_dict(torch.load(trained_weights))
        reload_model.cls_MPL.eval()

        gt = []
        pre = []
        for i, datax in enumerate(var_data):
            imgLeft= datax[0]
            imgRight= datax[1]
            joint_config = datax[2]
            objectPose = datax[3]

            imgs1= imgLeft.to(self.device)
            imgs2= imgRight.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            # print(cnt, imgs.shape)
            # print(imgs.is_cuda)
            outputs = reload_model(imgs1, imgs2, jc)

            labelx = labels.detach().cpu().numpy()
            outputx = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            gt.append(labelx)
            pre.append(outputx)


        return gt, pre




    
            

import time
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 采集数据
    m = TrainClassification()
    time.sleep(2)
    gt, predict = m.var(trained_weights = "./weights/cls_weight_LS.pth")
    # print(gt[0], type(gt[0]), type(gt))

    gtlist= []
    ylist = []
    for i in range(len(gt)):
        for j in range(len(gt[i])):
            gtlist.append(gt[i][j])
            ylist.append(predict[i][j])        

    gtlist = np.array(gtlist)
    ylist = np.array(ylist)
    print(gtlist.shape,ylist.shape, )

    cnt = 0
    for i in range(len(gtlist)):
        if gtlist[i] == ylist[i]:
            cnt += 1
    print(cnt/len(gtlist))

    confusion_matrix = metrics.confusion_matrix(gtlist,ylist, normalize="pred")
    print(confusion_matrix)
    # import pandas as pd
    # data_df = pd.DataFrame(confusion_matrix)
    # w = pd.ExcelWriter("1.xlsx")
    # data_df.to_excel(w)
    # w.save()
    
    Accuracy = metrics.accuracy_score(gtlist,ylist)
    print(Accuracy)

    #item_list = ["Ball"]
    item_list = ["Cylinder", "Square\nPrism", "Triangular\nPrism","Tube1","Tube2","Tube3","Tube4"]
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = item_list)

    cm_display.plot(include_values=True, values_format='.2g',colorbar=False, cmap=plt.cm.GnBu, xticks_rotation=45)
    plt.tight_layout()
    plt.show()
    
    # [[0.99196787 0.003003   0.         0.         0.         0.    0.       ]
    # [0.00401606 0.99399399 0.         0.         0.         0.    0.        ]
    # [0.         0.         0.9937565  0.         0.         0.    0.02270382]
    # [0.00100402 0.         0.         0.99897541 0.00620476 0.00700701    0.00206398]
    # [0.         0.         0.00312175 0.         0.99276112 0.    0.00619195]
    # [0.00301205 0.003003   0.         0.00102459 0.         0.99299299    0.        ]
    # [0.         0.         0.00312175 0.         0.00103413 0.    0.96904025]]

    


