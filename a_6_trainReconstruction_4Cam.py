# -*- coding:utf-8 -*-

"""
@Modified: 
@Description: Train autoencoder with additional data from behind camera
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
from AutoEncoder import AE
from LoadData_4Cam import ReconstDataset
import matplotlib.pyplot as plt
import os
import cv2
import csv
from datetime import datetime

torch.cuda.empty_cache()

class TrainEncoderDecoder(object):
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.momentum = 0.3
        self.epoch = 200
        #self.saved_pth = './weights/reconstruction_weight_LS_64_3bjects_240914.pth'#1/3
        #self.saved_pth = './weights/reconstruction_weight_PEACH_241227.pth'#1/3
        self.saved_pth = './weights/reconstruction_weight_J2_250801.pth'#1/3
        self.pre_weight = None
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # Loss function
        self.recons_loss = torch.nn.MSELoss()

        # Data paths for training and validation (include behind data)
        #self.data_folder_train = ["./IMG_DATA_LS/IMG_DATA_MASTERBALL_0830/train.txt","./IMG_DATA_LS/IMG_DATA_BOTTLE_0830/train.txt","./IMG_DATA_LS/IMG_DATA_PEACH_0830/train.txt"]  # #2/3
        #self.data_folder_var = ["./IMG_DATA_LS/IMG_DATA_MASTERBALL_0830/var.txt","./IMG_DATA_LS/IMG_DATA_BOTTLE_0830/var.txt","./IMG_DATA_LS/IMG_DATA_PEACH_0830/var.txt"]  # #3/3
        #self.data_folder_train = ["./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/train.txt","./IMG_DATA_LS/IMG_DATA_PEACH_0912/train.txt","./IMG_DATA_LS/IMG_DATA_COVER_0912/train.txt"]  # #2/3
        #self.data_folder_var = ["./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/var.txt","./IMG_DATA_LS/IMG_DATA_PEACH_0912/var.txt","./IMG_DATA_LS/IMG_DATA_COVER_0912/var.txt"]  # #3/3
        self.data_folder_train = ["./IMG_DATA_LS/IMG_DATA_J2_250801/train.txt"]  # #2/3
        self.data_folder_var = ["./IMG_DATA_LS/IMG_DATA_J2_250801/var.txt"]  # #3/3
             
        self.width = 320
        self.height = 320

    def train(self):
        # Load data
        traing_data = []
        for i in range(len(self.data_folder_train)):
            dataset_image = ReconstDataset(data_path=self.data_folder_train[i], scaled_width=self.width, scale_height=self.height)
            print("sample numbers: ", len(dataset_image))
            #print("dataset_image shape:", dataset_image.size) 
            temp = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
            traing_data.append(temp)

        # Initialize model
        reconstruction_model = AE(1, 1).to(self.device)
        
        if self.pre_weight is not None:
            reconstruction_model.load_state_dict(torch.load(self.pre_weight))

        # Optimizer
        optimizer = optim.Adam(reconstruction_model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        vloss = torch.nn.MSELoss()

        # Training loop
        pre_loss = 10000
        epoch_train_loss = []
        epoch_var_loss = []
        for epoch in range(self.epoch):
            reconstruction_model.train()
            running_loss = 0.0
            cnt = 0

            for k in range(len(traing_data)):
                for i, datax in enumerate(traing_data[k]):
                    imgLeft = datax[0]
                    imgRight = datax[1]
                    imgBehind = datax[2]  # 新增的处理 behind 数据
                    #print("imgLeft shape:", imgLeft.shape)  
                    #print("imgRight shape:", imgRight.shape)  
                    #print("imgBehind shape:", imgBehind.shape)  
                    imgs = torch.cat((imgLeft, imgRight, imgBehind), axis=0)  # 新增的 behind 数据拼接
                    #print("imgs shape:", imgs.shape)  

                    #print("imgs",len(imgs))
                    imgs = imgs.to(self.device)
                    labels = imgs

                    outputs = reconstruction_model(imgs)
                    loss = vloss(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    cnt += 1

            # Save loss
            epoch_train_loss.append(running_loss / cnt)
            print('=============train loss=============')
            print('epoch: %d, train loss: %.6f' % (epoch, running_loss/cnt), 'lr: ', get_lr(optimizer))

            if epoch == 0:
                torch.save(reconstruction_model.state_dict(), self.saved_pth)
                var_loss = self.val(model=reconstruction_model)
                epoch_var_loss.append(var_loss)
                print('============var_loss==============')
                print("epoch:", epoch, "var_loss:", var_loss)
                pre_loss = running_loss / cnt
            else:
                var_loss = self.val(model=reconstruction_model)
                epoch_var_loss.append(var_loss)
                print('============var_loss==============')
                print("epoch:", epoch, "var_loss:", var_loss)
                if running_loss / cnt < pre_loss:
                    pre_loss = running_loss / cnt
                    print('============save reconstruction_model==============')
                    torch.save(reconstruction_model.state_dict(), self.saved_pth)

        print('=============each epoch train loss, var_loss=============')
        np.savetxt('./weights/rw_trainLoss_' + datetime.now().strftime("%m%d-%H%M%S") + '.txt', epoch_train_loss)
        np.savetxt('./weights/rw_varLoss_' + datetime.now().strftime("%m%d-%H%M%S") + '.txt', epoch_var_loss)

    @torch.no_grad()
    def val(self, model=None, trained_weights="./weights/reconstruction_weight.pth"):
        # Load validation data
        var_data = []
        for i in range(len(self.data_folder_var)):
            dataset_image = ReconstDataset(data_path=self.data_folder_var[i], scaled_width=self.width, scale_height=self.height)
            print("sample numbers: ", len(dataset_image))
            temp = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
            var_data.append(temp)        
        
        # Load model
        if model is None:
            reload_model = AE(1, 1).to(self.device)
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        vloss = torch.nn.MSELoss()
        reload_model.eval()
        running_loss = 0.0
        cnt = 0

        for k in range(len(var_data)):
            for i, datax in enumerate(var_data[k]):
                imgLeft = datax[0]
                imgRight = datax[1]
                imgBehind = datax[2]  # 新增的处理 behind 数据

                imgs = torch.cat((imgLeft, imgRight, imgBehind), axis=0)  # 新增的 behind 数据拼接
                imgs = imgs.to(self.device)
                labels = imgs

                outputs = reload_model(imgs)
                loss = vloss(outputs, labels)
                running_loss += loss.item()
                cnt += 1

        return running_loss / cnt

    
def get_lr(opti):
    for param_group in opti.param_groups:
        return param_group['lr']
                
if __name__ == "__main__":
    test = TrainEncoderDecoder()
    test.train()

    print('============final var loss==============')
