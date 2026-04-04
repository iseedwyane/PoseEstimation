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
from PoseNet import PNet, EncordFun
from LoadData import ReconstDataset

from scipy.spatial.transform import Rotation as spR


class TrainRegression(object):
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.momentum = 0.3
        self.epoch = 500
        self.pre_model_pth = "./weights/reconstruction_weight.pth"
        self.saved_pth = "./weights/pose_weight.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        # loss
        self.recons_loss =torch.nn.MSELoss()
        self.data_folder = "./IMG_DATA_cycle/train.txt"
        self.data_folder_var = "./IMG_DATA_cycle/var.txt"
        self.width = 320
        self.height = 320

    
    def train(self):
        # load data
        dataset_image = ReconstDataset(data_path=self.data_folder, scaled_width = self.width, scale_height= self.height)
        print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
        
        pose_model = PNet(self.pre_model_pth).to(self.device)
        # optimizer
        # optimizer = optim.SGD(pose_model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        optimizer = optim.Adam(pose_model.parameters(), lr=self.learning_rate)
        vloss = torch.nn.MSELoss()

        for epoch in range(self.epoch):
            # training
            pose_model.train()
            running_loss = 0.0
            for i, datax in enumerate(traing_data):
                imgLeft= datax[0]
                imgRight= datax[1]
                joint_config = datax[2]
                objectPose = datax[3]

                imgs1= imgLeft.to(self.device)
                imgs2= imgRight.to(self.device)
                jc = joint_config.to(self.device)
                labels = objectPose.to(self.device)
                labels = labels.squeeze(1)

                # print(imgs.is_cuda)
                # print("device:", self.device)
                outputs = pose_model(imgs1, imgs2, jc)
                # print('batch size: ', i, imgs.shape, len(labels))
                # print('labels: ', labels.shape)
                # print('outputs: ', outputs.shape)
                # loss = vloss(outputs, labels)
                loss = loss_geo(outputs, labels)
                # print("epoch %d, batch %d," % (epoch, i), 'loss: ', loss.item())
                
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                if i % 20 == 19:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/20))
                    running_loss = 0.0


        # save weight
        torch.save(pose_model.state_dict(), self.saved_pth)
        # torch.save(reconstruction_model, self.saved_pth)

    @torch.no_grad()
    def var(self, model = None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=self.data_folder_var, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        if model==None:
            reload_model = PNet(self.pre_model_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        vloss = torch.nn.MSELoss()
        # 
        reload_model.eval()
            
        running_loss = 0.0
        for i, datax in enumerate(traing_data):
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
            loss = vloss(outputs, labels)
            running_loss += loss.item()
        # print("var loss:", running_loss/len(traing_data))
        return running_loss/len(traing_data)
    
    @torch.no_grad()
    def varGeo(self, model = None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=self.data_folder_var, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        if model==None:
            reload_model = PNet(self.pre_model_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        vloss = torch.nn.MSELoss()
        # 
        reload_model.eval()
            
        total_transation_err = 0.0
        total_rotation_err = 0.0
        for i, datax in enumerate(traing_data):
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
            # print(outputs[])
            # print(outputs.detach().cpu().numpy())
            transation_err, rotation_err = self.measureDistance(outputs, labels)
            total_transation_err += transation_err
            total_rotation_err += rotation_err

        # print("var loss:", running_loss/len(traing_data))
        return total_transation_err/len(traing_data), total_rotation_err/len(traing_data)
    
    # Sattler T, Benchmarking 6dof outdoor visual localization in changing conditions, CVPR, 2018
    def measureDistance(self, predict_pose, true_pose):
        loss = torch.nn.MSELoss()

        predict_pose = predict_pose.detach().cpu().numpy()
        true_pose = true_pose.detach().cpu().numpy()
        if len(predict_pose) != len(true_pose):
            raise Exception("predice and label size are not equal!")

        transitionError = []
        rotationError = []
        for i in range(len(predict_pose)):
            
            temp_trans = np.linalg.norm(predict_pose[i,0:3]-true_pose[i,0:3])

            temp_transfer1 = spR.from_rotvec(predict_pose[i,3:6])
            temp_transfer2 = spR.from_rotvec(true_pose[i,3:6])
            predict_matrix = temp_transfer1.as_matrix()
            true_matrix = temp_transfer2.as_matrix()

            err_matrix = np.dot(np.linalg.inv(true_matrix), predict_matrix)
            angle = np.abs(np.arccos((np.trace(err_matrix)-1)/2))

            transitionError.append(temp_trans)
            rotationError.append(angle)
        
        return np.mean(transitionError), np.mean(rotationError)
 
def loss_geo(output, target):
    tx = torch.norm(output[:,0:3]-target[:,0:3], dim=1)
    tx = torch.mean(tx)
    rx = torch.norm(output[:,3:6]-target[:,3:6], dim=1)
    rx = torch.mean(rx)
    return 1000*tx


if __name__ == "__main__":
    test = TrainRegression()
    test.train()
    # x = test.var(trained_weights = "./weights/pose_weight.pth")
    # x = test.varGeo(trained_weights = "./weights/pose_weight.pth")
    # print(x)

