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
from LoadData_Welding import ReconstDataset
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
        self.saved_pth = './weights/reconstruction_weight_welding.pth'#1/3
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
        self.data_folder_train = ["./IMG_DATA_LS/DataforSVAE/train.txt"]  # #2/3
        self.data_folder_var = ["./IMG_DATA_LS/DataforSVAE/var.txt"]  # #3/3       
        
        self.width = 320
        self.height = 320

    def train(self):
        # load data
        traing_data = []
        for i in range(len(self.data_folder_train)):
            dataset_image = ReconstDataset(data_path=self.data_folder_train[i], scaled_width = self.width, scale_height= self.height)
            print("sample numbers: ", len(dataset_image))
            temp = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
            traing_data.append(temp)




        reconstruction_model = AE(1, 1).to(self.device)
#        if self.device.type != 'cpu':
#            reconstruction_model = torch.nn.DataParallel(reconstruction_model, #device_ids=[0,1,2,3])
        
        if self.pre_weight != None:
            reconstruction_model.load_state_dict(torch.load(self.pre_weight))

        # optimizer
        # optimizer = optim.SGD(reconstruction_model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        optimizer = optim.Adam(reconstruction_model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        vloss = torch.nn.MSELoss()

        # training
        
        pre_loss = 10000
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epoch/2), eta_min=0.0005)

        epoch_train_loss = []
        epoch_var_loss = []
        for epoch in range(self.epoch):
            reconstruction_model.train()
            running_loss = 0.0
            cnt = 0

            for k in range(len(traing_data)):
                for i, datax in enumerate(traing_data[k]):
                    imgLeft= datax[0]

                    imgs= torch.cat((imgLeft),axis = 0)
                    imgs = imgs.to(self.device)
                    labels = imgs
                    # print(imgs.is_cuda)
                    # print("device:", self.device)
                    outputs = reconstruction_model(imgs)

                    loss = vloss(outputs, labels)
                    # print("epoch %d, dataset %d, batch %d" % (epoch, k, i), 'loss: ', loss.item(), 'lr: ', get_lr(optimizer))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    cnt += 1

            # save loss
            epoch_train_loss.append(running_loss/cnt)
            # scheduler.step()
            #训练过程中的日志信息：每个 epoch 的损失和学习率
            print('=============train loss=============')
            print('epoch: %d, train loss: %.6f' % (epoch, running_loss/cnt), 'lr: ', get_lr(optimizer))
            # save weight
            if epoch == 0:
                torch.save(reconstruction_model.state_dict(), self.saved_pth)
                # var_loss = self.val(self.saved_pth)
                var_loss = self.val(model=reconstruction_model)
                epoch_var_loss.append(var_loss)
                #验证集的损失：用于评估模型在验证集上的表现
                print('============var_loss==============')
                print("epoch:", epoch, "var_loss:", var_loss)
                pre_loss = running_loss/cnt
            else:
                # var_loss = self.val(self.saved_pth)
                var_loss = self.val(model=reconstruction_model)
                epoch_var_loss.append(var_loss)
                #验证集的损失：用于评估模型在验证集上的表现
                print('============var_loss==============')
                print("epoch:", epoch, "var_loss:", var_loss)
                if running_loss/cnt<pre_loss:
                    pre_loss = running_loss/cnt
                    print('============save reconstruction_model==============')
                    torch.save(reconstruction_model.state_dict(), self.saved_pth)
        #each epoch的训练和验证损失值
        print('=============each epoch train loss, var_loss=============')    
        np.savetxt('./weights/rw_trainLoss.txt'+ datetime.now().strftime("%m%d-%H%M%S"), epoch_train_loss)
        np.savetxt('./weights/rw_varLoss.txt'+ datetime.now().strftime("%m%d-%H%M%S"), epoch_var_loss)
        # save weight
        # torch.save(reconstruction_model.state_dict(), self.saved_pth)
        # torch.save(reconstruction_model, self.saved_pth)
        # torch.save(reconstruction_model.module.state_dict(), self.saved_pth)

    def showTraingIteration(self, model):
        for name, parms in model.named_parameters():
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        print('==========================')

    @torch.no_grad()
    def val(self, model = None, trained_weights = "./weights/reconstruction_weight.pth"):
        # # 载入测试集
        # dataset_image = ReconstDataset(data_path=self.data_folder_var[0], scaled_width = self.width, scale_height= self.height)
        # # print("sample numbers: ", len(dataset_image))
        # traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
        
        # load data
        var_data = []
        for i in range(len(self.data_folder_var)):
            dataset_image = ReconstDataset(data_path=self.data_folder_var[i], scaled_width = self.width, scale_height= self.height)
            print("sample numbers: ", len(dataset_image))
            temp = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
            var_data.append(temp)        
        
        # 载入模型
        if model==None:
            reload_model = AE(1, 1).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
            #reload_model.load_state_dict(torch.load(trained_weights, map_location=self.device))
        else:
            reload_model = model

        vloss = torch.nn.MSELoss()
        # 
        reload_model.eval()
        running_loss = 0.0
        cnt = 0
        for k in range(len(var_data)):
            for i, datax in enumerate(var_data[k]):
                imgLeft= datax[0]
                imgRight= datax[1]

                imgs= torch.cat((imgLeft, imgRight),axis = 0)
                imgs = imgs.to(self.device)
                labels = imgs

                outputs = reload_model(imgs)

                loss = vloss(outputs, labels)
                # print("epoch %d, dataset %d, batch %d" % (epoch, k, i), 'loss: ', loss.item(), 'lr: ', get_lr(optimizer))
                running_loss += loss.item()
                cnt += 1

        # for i, datax in enumerate(traing_data):
        #     imgLeft= datax[0]
        #     imgRight= datax[1]

        #     # imgs = imgLeft.to(self.device)
        #     imgs= torch.cat((imgLeft, imgRight),axis = 0)
        #     imgs = imgs.to(self.device)
        #     labels = imgs

        #     # print(cnt, imgs.shape)
        #     # print(imgs.is_cuda)
        #     outputs = reload_model(imgs)
        #     loss = vloss(outputs, labels)
        #     running_loss += loss.item()
        # print("var loss:", running_loss/len(traing_data))
        # return running_loss/len(traing_data)
        return running_loss/cnt
            


    @torch.no_grad()
    def val_img(self, trained_weights = "./weights/reconstruction_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=self.data_folder_var[0], scaled_width = self.width, scale_height= self.height)
        print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        reload_model = AE(1, 1).to(self.device)
        # if self.device.type != 'cpu':
        #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
        
        reload_model.load_state_dict(torch.load(trained_weights))

        vloss = torch.nn.MSELoss()
        # 
        reload_model.eval()
            
        running_loss = 0.0
        cnt = 0
        for i, datax in enumerate(traing_data):
            imgLeft= datax[0]
            imgRight= datax[1]

            imgs = imgLeft.to(self.device)
            # imgs= torch.cat((imgLeft, imgRight),axis = 0)
            # imgs = imgs.to(self.device)
            # labels = imgs

            # print(cnt, imgs.shape)
            # print(imgs.is_cuda)
            outputs = reload_model(imgs)
            # plt.imshow(outputs[0].permute(1, 2, 0))
            
            a = torch.cat((imgs, outputs),axis = 3)
            # print(outputs[0][0])
            cv2.imshow("a", (a[0][0]).cpu().detach().numpy())
            cv2.waitKey()

            # loss = vloss(outputs, labels)
            # running_loss += loss.item()
            if cnt > 2:
                break
            cnt += 1


    
def get_lr(opti):
    for param_group in opti.param_groups:
        return param_group['lr']
                
if __name__ == "__main__":
    test = TrainEncoderDecoder()
    test.train()
    # test.val_img("./weights/reconstruction_weight_128.pth")
    
    #l = test.val(trained_weights="./weights/reconstruction_weight_64.pth")

    ##最终的验证损失：在整个训练完成后，打印最终的验证损失。
    print('============final var loss==============')
    #print('loss:', l)


    # x = AE(1,1)
    # print(x._encoder.enc_conv0.weight)
    # model.conv1.bias = nn.Parameter(torch.load("state_dict_model.pt")['conv1.bias'])
    # for m in x._encoder.parameters():
    #     print(m)
    #ls

    #torch.save(reconstruction_model.module.state_dict(), self.saved_pth)
