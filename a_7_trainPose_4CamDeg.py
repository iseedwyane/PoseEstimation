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
from PoseNet_4Cam import PNet, EncordFun
from LoadData_4Cam import ReconstDataset
from torchvision import transforms, utils
import os
from scipy.spatial.transform import Rotation as spR
import cv2
import copy
import random
from datetime import datetime

def angle_difference_loss(pred, true): #0922New loss
    diff = torch.abs(pred - true)
    min_diff = torch.minimum(diff, 360 - diff)  # 确保周期性处理
    return torch.mean(min_diff)

class TrainRegression(object):
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.momentum = 0.3
        self.epoch = 200#5/5
        self.pre_model_pth = "./weights/reconstruction_weight_VASE_250107.pth"  # 1/4
        self.saved_pth = "./weights/pose_weight.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # loss
        self.recons_loss = torch.nn.MSELoss()
        self.data_folder = "./IMG_DATA"
        self.data_folder_var = "./IMG_DATA"
        print("============data_folder_var_invaild", self.data_folder_var, "============")
        self.width = 320
        self.height = 320

    def augment(self, data):
        probability = random.uniform(0, 1)
        if probability < 0.4:
            return data
        elif 0.4 <= probability < 0.8:
            jitter = transforms.ColorJitter(brightness=.4, contrast=.3, saturation=0.2)
            trans_data = [jitter(data[i]) for i in range(len(data))]
            return torch.stack(trans_data)
        else:
            blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5))
            trans_data = [blurrer(data[i]) for i in range(len(data))]
            return torch.stack(trans_data)
        

    
    def train(self):
        # load data
        dataset_image = ReconstDataset(data_path=self.data_folder, scaled_width=self.width, scale_height=self.height)
        print("sample numbers: ", len(dataset_image)) #sample numbers:  3462 = 4328 (0907folder)*0.8

        print("sample dataset_image[3]'s len: ", len(dataset_image[3])) #sample dataset_image[3]'s len:  5, 
        #5 tensor, is return torch.Tensor(np.array([img1_data])), torch.Tensor(np.array([img2_data])), torch.Tensor([img3_data]), torch.Tensor(np.array([joint_config])), torch.Tensor(np.array([objectPose]))
 


        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=False, collate_fn=None)

        pose_model = PNet(self.pre_model_pth).to(self.device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, pose_model.parameters()), lr=self.learning_rate)
        vloss = torch.nn.MSELoss()
        angleloos = torch.nn.L1Loss()

        pre_loss = 10e10
        epoch_train_loss = []
        epoch_var_loss = []

        for epoch in range(self.epoch):
            # training
            pose_model.pose_MPL.train()
            running_loss = 0.0
            cnt = 0
            for i, datax in enumerate(traing_data):
                imgLeft = datax[0]
                imgRight = datax[1]
                imgBehind = datax[2]  # 新增的处理 behind 数据
                joint_config = datax[3]
                objectPose = datax[4]

                imgs1 = imgLeft.to(self.device)
                imgs2 = imgRight.to(self.device)
                imgs3 = imgBehind.to(self.device)
                jc = joint_config.to(self.device)
                labels = objectPose.to(self.device)
                labels = labels.squeeze(1)

                optimizer.zero_grad()

                outputs = pose_model(imgs1, imgs2, imgs3, jc)  # 处理 behind 数据
                l1 = vloss(outputs[:, 0:3], labels[:, 0:3])
                #l2 = vloss(outputs[:, 3:6], labels[:, 3:6])
                # 使用自定义角度差损失
               
                l_rot_x = angleloos(outputs[:, 3], labels[:, 3])
                l_rot_y = angleloos(outputs[:, 4], labels[:, 4])
                l_rot_z = angleloos(outputs[:, 5], labels[:, 5])
                loss = 0.1 * l1 + 10 * (l_rot_x + l_rot_y )+ 1*l_rot_z
                #loss = 0.01 * l1 + 0.001 * l2#1/2
                #loss = 0.01 * l1 + 0.001 * l3 + 0.001*l4+0.001*l5#1/2
                # print('outputs:  ', outputs)
                # print('labels:  ', labels)
                # print('loss: l1 ', l1)
                # print('loss: l3 ', l_rot_x)
                # print('loss: l4 ', l_rot_y)
                # print('loss: l5 ', l_rot_z)
                # print('loss: ', loss)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                cnt += 1

            epoch_train_loss.append(running_loss / cnt)
            epoch_var_loss.append(self.var(self.data_folder_var, model=pose_model))

            print('epoch: %d, train loss: %.6f, pre_loss: %.6f' % (epoch, running_loss / cnt, pre_loss))
            if running_loss / cnt < pre_loss:
                pre_loss = running_loss / cnt
                print("======save weights======", "epoch:", epoch)
                torch.save(pose_model.state_dict(), self.saved_pth)

        np.savetxt('./weights/pw_trainLoss.txt' + datetime.now().strftime("%m%d-%H%M%S") + '.txt', epoch_train_loss)
        np.savetxt('./weights/pw_varLoss.txt' + datetime.now().strftime("%m%d-%H%M%S") + '.txt', epoch_var_loss)

    @torch.no_grad()
    def var(self, varData, model=None, trained_weights="./weights/pose_weight.pth"):
        dataset_image = ReconstDataset(data_path=varData, scaled_width=self.width, scale_height=self.height)
        var_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)

        if model is None:
            reload_model = PNet(self.saved_pth).to(self.device)
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        reload_model.pose_MPL.eval()
        vloss = torch.nn.MSELoss()
        angleloos = torch.nn.L1Loss()

        running_loss = 0.0
        cnt = 0
        for i, datax in enumerate(var_data):
            imgLeft = datax[0]
            imgRight = datax[1]
            imgBehind = datax[2]
            joint_config = datax[3]
            objectPose = datax[4]

            imgs1 = imgLeft.to(self.device)
            imgs2 = imgRight.to(self.device)
            imgs3 = imgBehind.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            outputs = reload_model(imgs1, imgs2, imgs3, jc)
            l1 = vloss(outputs[:, 0:3], labels[:, 0:3])
            #l2 = vloss(outputs[:, 3:6], labels[:, 3:6])
            # 使用自定义角度差损失
            
            l_rot_x = angleloos(outputs[:, 3], labels[:, 3])
            l_rot_y = angleloos(outputs[:, 4], labels[:, 4])
            l_rot_z = angleloos(outputs[:, 5], labels[:, 5])
            loss = 0.1 * l1 + 10 * (l_rot_x + l_rot_y )+ 1*l_rot_z
            running_loss += loss.item()
            cnt += 1
        return running_loss / cnt

    @torch.no_grad()
    def varGeo(self, varData, model=None, trained_weights="./weights/pose_weight.pth"):
        dataset_image = ReconstDataset(data_path=varData, scaled_width=self.width, scale_height=self.height)
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)

        if model is None:
            reload_model = PNet(self.saved_pth).to(self.device)
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        reload_model.pose_MPL.eval()
        total_transation_err = 0.0
        total_rotation_err = 0.0

        for i, datax in enumerate(traing_data):
            imgLeft = datax[0]
            imgRight = datax[1]
            imgBehind = datax[2]
            joint_config = datax[3]
            objectPose = datax[4]

            imgs1 = imgLeft.to(self.device)
            imgs2 = imgRight.to(self.device)
            imgs3 = imgBehind.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            outputs = reload_model(imgs1, imgs2, imgs3, jc)
            transation_err, rotation_err = self.measureDistance(outputs, labels)
            total_transation_err += transation_err
            total_rotation_err += rotation_err

        return total_transation_err / len(traing_data), total_rotation_err / len(traing_data) * 180.0 / math.pi

    def measureDistance(self, predict_pose, true_pose):
        predict_pose = predict_pose.detach().cpu().numpy()
        true_pose = true_pose.detach().cpu().numpy()
        transitionError = []
        rotationError = []

        for i in range(len(predict_pose)):
            temp_trans = np.linalg.norm(predict_pose[i, 0:3] - true_pose[i, 0:3])

            # 将度数转换为弧度
            outputs_rad = np.deg2rad(predict_pose[i, 3:6])
            labels_rad = np.deg2rad(true_pose[i, 3:6])

            temp_transfer1 = spR.from_rotvec(outputs_rad)
            temp_transfer2 = spR.from_rotvec(labels_rad)
            predict_matrix = temp_transfer1.as_matrix()
            true_matrix = temp_transfer2.as_matrix()

            err_matrix = np.dot(np.linalg.inv(true_matrix), predict_matrix)
            angle = np.abs(np.arccos((np.trace(err_matrix) - 1) / 2))

            transitionError.append(temp_trans)
            #rotationError.append(np.rad2deg(angle))  # 转换为度数
            rotationError.append(angle)  

        return np.mean(transitionError), np.mean(rotationError)

    @torch.no_grad()
    def varGeoList(self, varData, model=None, trained_weights="./weights/pose_weight.pth"):
        dataset_image = ReconstDataset(data_path=varData, scaled_width=self.width, scale_height=self.height)
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)

        if model is None:
            reload_model = PNet(self.saved_pth).to(self.device)
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        reload_model.pose_MPL.eval()
        total_transation_err = np.array([])
        total_rotation_err = np.array([])

        for i, datax in enumerate(traing_data):
            imgLeft = datax[0]
            imgRight = datax[1]
            imgBehind = datax[2]
            joint_config = datax[3]
            objectPose = datax[4]

            imgs1 = imgLeft.to(self.device)
            imgs2 = imgRight.to(self.device)
            imgs3 = imgBehind.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            outputs = reload_model(imgs1, imgs2, imgs3, jc)
            transation_err, rotation_err = self.measureDistanceList(outputs, labels)
            total_transation_err = np.hstack((total_transation_err, transation_err))
            total_rotation_err = np.hstack((total_rotation_err, rotation_err))

        return total_transation_err, total_rotation_err

    @torch.no_grad()
    def varGeoListDimension(self, varData, model=None, trained_weights="./weights/pose_weight.pth"):
        dataset_image = ReconstDataset(data_path=varData, scaled_width=self.width, scale_height=self.height)
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)

        if model is None:
            reload_model = PNet(self.saved_pth).to(self.device)
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        reload_model.pose_MPL.eval()
        total_pre = 0
        total_gt = 0

        for i, datax in enumerate(traing_data):
            imgLeft = datax[0]
            imgRight = datax[1]
            imgBehind = datax[2]
            joint_config = datax[3]
            objectPose = datax[4]

            imgs1 = imgLeft.to(self.device)
            imgs2 = imgRight.to(self.device)
            imgs3 = imgBehind.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            outputs = reload_model(imgs1, imgs2, imgs3, jc)

            predict_pose = outputs.detach().cpu().numpy()
            true_pose = labels.detach().cpu().numpy()

            if type(total_gt) == int:
                total_pre = predict_pose
                total_gt = true_pose
            else:
                total_pre = np.vstack((total_pre, predict_pose))
                total_gt = np.vstack((total_gt, true_pose))

        return total_pre, total_gt

    def measureDistanceList(self, predict_pose, true_pose):
        predict_pose = predict_pose.detach().cpu().numpy()
        true_pose = true_pose.detach().cpu().numpy()
        transitionError = []
        rotationError = []

        for i in range(len(predict_pose)):
            temp_trans = np.linalg.norm(predict_pose[i, 0:3] - true_pose[i, 0:3])

            # 将度数转换为弧度
            outputs_rad = np.deg2rad(predict_pose[i, 3:6])
            labels_rad = np.deg2rad(true_pose[i, 3:6])

            temp_transfer1 = spR.from_rotvec(outputs_rad)
            temp_transfer2 = spR.from_rotvec(labels_rad)
            predict_matrix = temp_transfer1.as_matrix()
            true_matrix = temp_transfer2.as_matrix()

            err_matrix = np.dot(np.linalg.inv(true_matrix), predict_matrix)
            angle = np.abs(np.arccos((np.trace(err_matrix) - 1) / 2))

            transitionError.append(temp_trans)
            #rotationError.append(np.rad2deg(angle))  # 转换为度数
            rotationError.append(angle)

        return np.array(transitionError), np.array(rotationError)




# Sattler T, Benchmarking 6dof outdoor visual localization in changing conditions, CVPR, 2018
def rotationError(estimate_matrix, target_matrix):
    err_matrix = torch.inverse(target_matrix) @ estimate_matrix
    alpha = torch.abs(torch.acos((torch.trace(err_matrix) - 1) / 2))
    return alpha

if __name__ == "__main__":
   # saved_path = ["./weights/pose_weight_masterball_0923.pth","./weights/pose_weight_peach_0923.pth","./weights/pose_weight_cover_0923.pth"]#2/4 output
    #data_path = ["IMG_DATA_MASTERBALL_0907","IMG_DATA_PEACH_0912","IMG_DATA_COVER_0912"]#3/4 input
    #error_list = ["error_masterball_0923","error_peach_0923","error_cover_0923"]#4/4 output
    saved_path = ["./weights/pose_weight_vase_250108.pth"]#2/4
    data_path = ["IMG_DATA_VASE_250107"]#3/4
    error_list = ["error_vase_250108"]#4/4


    test = TrainRegression()
    for i in range(len(saved_path)):
        test.saved_pth = saved_path[i]
        test.data_folder = "./IMG_DATA_LS/" + data_path[i] + "/train.txt"
        test.data_folder_var = "./IMG_DATA_LS/" + data_path[i] + "/all.txt"

        print("=================", data_path[i], "===================")
        test.train()
        x1 = test.var(test.data_folder, trained_weights=test.saved_pth)
        x2 = test.varGeo(test.data_folder, trained_weights=test.saved_pth)
        print("train:", x1, x2)
        x1 = test.var(test.data_folder_var, trained_weights=test.saved_pth)
        x2 = test.varGeo(test.data_folder_var, trained_weights=test.saved_pth)
        print("var:", x1, x2)

        # Set the new path for saving error files
        result_folder = f"./IMG_DATA_LS/{data_path[i]}/Results/"
        # Create Results directory if it doesn't exist
        os.makedirs(result_folder, exist_ok=True)


        file_name_t = result_folder + error_list[i] + "_translation.txt"
        file_name_r = result_folder + error_list[i] + "_rotation.txt"

        # translaton and orientation error        
        x2 = test.varGeoList(test.data_folder_var, trained_weights = test.saved_pth)
        print("var:", np.mean(x2[0]), np.mean(x2[1])*180/math.pi)
        np.savetxt(file_name_t, x2[0])
        np.savetxt(file_name_r, x2[1])

        file_name_t = result_folder + error_list[i] + "_pre.txt"
        file_name_r = result_folder + error_list[i] + "_gt.txt"
        a,b = test.varGeoListDimension(test.data_folder_var, trained_weights = test.saved_pth)

        np.savetxt(file_name_t, a)
        np.savetxt(file_name_r, b)

        file_name_all = result_folder + error_list[i] + "_all.txt"

        a,b = test.varGeoListDimension(test.data_folder_var, trained_weights = test.saved_pth)
        np.savetxt(file_name_all, b)
        
