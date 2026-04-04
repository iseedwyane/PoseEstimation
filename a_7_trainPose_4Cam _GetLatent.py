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
from torchvision import transforms,utils

from scipy.spatial.transform import Rotation as spR
import cv2
import copy
import random
from datetime import datetime
import os

class TrainRegression(object):
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.momentum = 0.3
        self.epoch = 200
        self.pre_model_pth = "./weights/reconstruction_weight_VASE_250111.pth"#1/4
        self.saved_pth = "./weights/pose_weight.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        # loss
        self.recons_loss =torch.nn.MSELoss()
        self.data_folder = "./IMG_DATA"#any path is ok, because training don't need this input.
        self.data_folder_var = "./IMG_DATA"
        print("============data_folder_var_invaild",self.data_folder_var,"============")
        self.width = 320
        self.height = 320

    def augment(self, data):
        probability = random.uniform(0,1)
        if probability<0.4:
            return data
        elif probability>=0.4 and probability <0.8:
            jitter = transforms.ColorJitter(brightness=.4, contrast=.3, saturation=0.2)
            trans_data = [jitter(data[i]) for i in range(len(data))]
            return torch.stack(trans_data)
        elif probability>=0.8:
            blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5))
            trans_data = [blurrer(data[i]) for i in range(len(data))]
            return torch.stack(trans_data)

    
    def train(self):
        # load data
        dataset_image = ReconstDataset(data_path=self.data_folder, scaled_width = self.width, scale_height= self.height)
        print("sample numbers: ", len(dataset_image))
        print("sample dataset_image[3]'s len: ", len(dataset_image[3]))

        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=False, collate_fn=None)
        
        pose_model = PNet(self.pre_model_pth).to(self.device)
        # optimizer
        # optimizer = optim.SGD(pose_model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        # optimizer = optim.Adam(pose_model.parameters(), lr=self.learning_rate)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, pose_model.parameters()), lr=self.learning_rate)
        vloss = torch.nn.MSELoss()

        pre_loss = 10e10
        epoch_train_loss = []
        epoch_var_loss = []
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300], gamma=0.5)
        for epoch in range(self.epoch):
            # training
            pose_model.pose_MPL.train()
            # pose_model.train()
            running_loss = 0.0
            cnt = 0
            for i, datax in enumerate(traing_data):
                imgLeft= datax[0]
                imgRight= datax[1]
                imgBehind = datax[2]  # 新增的处理 behind 数据 #LiS240826 behind
                #print("imgLeft shape:", imgLeft.shape)  
                #print("imgRight shape:", imgRight.shape)    
                #print("imgBehind shape:", imgBehind.shape)                 
                joint_config = datax[3]
                objectPose = datax[4]
                #print('joint_config: ', joint_config.shape, joint_config.get_device())
                #print('objectPose: ', objectPose.shape, objectPose.get_device())
                # # # augmentation
                # imgLeft = self.augment(imgLeft)
                # imgRight = self.augment(imgRight)

                imgs1= imgLeft.to(self.device)
                imgs2= imgRight.to(self.device)
                imgs3= imgBehind.to(self.device)#LiS240826 behind
                jc = joint_config.to(self.device)
                labels = objectPose.to(self.device)
                labels = labels.squeeze(1)

                # 对 force_tensor 进行归一化 #250110
                #scale_pose = [0.6, 0, 0.2, 2, 2, 1]  # 归一化比例 #1/2
                #for i in range(6):  # 对每一列进行归一化
                #    labels[:, i] = labels[:, i] / scale_pose[i]

                optimizer.zero_grad()

                outputs = pose_model(imgs1, imgs2, imgs3, jc)#LiS240826 behind
                # loss = loss_geo(outputs, labels)
                # loss = vloss(outputs, labels)
                l1 = vloss(outputs[:,0:3], labels[:,0:3])
                l2 = vloss(outputs[:,3:6], labels[:,3:6])
                #print('labels: ', labels.shape, labels.get_device())
                #print('outputs: ', outputs.shape, outputs.get_device())
                #print("outputs[:, 3:6] shape:", outputs[:, 3:6].shape)
                #print("labels[:, 3:6] shape:", labels[:, 3:6].shape)

                l_rx = vloss(outputs[:, 3], labels[:, 3])
                l_ry = vloss(outputs[:, 4], labels[:, 4])
                l_rz = vloss(outputs[:, 5], labels[:, 5])
                #loss = 1 * l1 + 0.5 * (l_rx + l_ry )+ 1*l_rz

                #loss = 0.1*l1 + 10*l2
                loss = l1 + l2
                #print('loss: l1 ', l1)
                #print('loss: l2 ', l2)
                #print('loss:  ', loss)
                # running_loss += loss.item()

                # if torch.any(torch.isnan(outputs)):
                #     print('xx:',imgs1,imgs2,jc)
                #     print('yy:', outputs)
                #     for k in range(len(imgs1)):
                #         cv2.imshow("a", (imgs1[k][0]).cpu().detach().numpy())
                #         cv2.waitKey()
                #     exit()33w
                # print('batch size: ', i, imgs1.shape, len(labels))
                #print('labels: ', labels.shape, labels.get_device())
                # print('outputs: ', outputs.shape, outputs.get_device())
                # print("epoch %d, batch %d," % (epoch, i), 'loss: ', loss.item())
                
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=pose_model.parameters(),max_norm=10)
                optimizer.step()
                # print("loss:",loss, loss.is_leaf, loss.grad)
                

                running_loss += loss.item()
                cnt += 1
                # if i % 20 == 19:
                #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/20))
                #     running_loss = 0.0
            # scheduler.step()

            # save loss
            # epoch_train_loss.append(running_loss/cnt)
            # if epoch%50==0:
            #     pth_name = "./weights/PoseWeight_0627/pose_weight_%s.pth"%epoch
            #     torch.save(pose_model.state_dict(), pth_name)
            #     print('epoch: %d, train loss: %.6f' % (epoch, running_loss/cnt))
            # 在每个 epoch 后计算平均训练损失并保存
            epoch_train_loss.append(running_loss / cnt) #LiS240826ADD  
            epoch_var_loss.append(self.var(self.data_folder_var, model=pose_model))#LiS240826ADD  

            print('epoch: %d, train loss: %.6f, pre_loss: %.6f' % (epoch, running_loss/cnt, pre_loss))
            if running_loss/cnt<pre_loss:
                pre_loss = running_loss/cnt
                print("======save weights======", "epoch:",epoch)
                torch.save(pose_model.state_dict(), self.saved_pth)

            # # save weight
            # if epoch == 0:
            #     torch.save(pose_model.state_dict(), self.saved_pth)
            #     pre_loss = running_loss/cnt
            # # elif epoch%10==0:
            # else:
            #     # var_loss = self.var(self.data_folder_var ,model=pose_model)
            #     var_loss = self.var(self.data_folder, model=pose_model)
            #     print("epoch:", epoch, ", var_loss:", var_loss)
            #     if var_loss<pre_loss:
            #         pre_loss = var_loss
            #         print("======save weights======", "epoch:",epoch)
            #         torch.save(pose_model.state_dict(), self.saved_pth)

        # # save weight
        np.savetxt('./weights/pw_trainLoss.txt'+ datetime.now().strftime("%m%d-%H%M%S") + '.txt', epoch_train_loss)
        np.savetxt('./weights/pw_varLoss.txt'+ datetime.now().strftime("%m%d-%H%M%S") + '.txt', epoch_var_loss)

    @torch.no_grad()
    def var(self, varData , model = None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=varData, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        var_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        if model==None:
            # reload_model = PNet(self.pre_model_pth).to(self.device)
            reload_model = PNet(self.saved_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
            
        else:
            # reload_model = copy.deepcopy(model)
            reload_model = model


        # reload_model.eval()
        reload_model.pose_MPL.eval()

        # # print(reload_model.pose_MPL)
        # for child in reload_model.pose_MPL.children():
        #     if type(child)==torch.nn.BatchNorm1d:
        #         child.track_running_stats = False
        # # print(reload_model.pose_MPL)
        

        vloss = torch.nn.MSELoss()
        # 
        running_loss = 0.0
        cnt = 0
        for i, datax in enumerate(var_data):
            imgLeft= datax[0]
            imgRight= datax[1]
            imgBehind = datax[2]  # 新增的处理 behind 数据 #LiS240826 behind
            joint_config = datax[3]
            objectPose = datax[4]

            imgs1= imgLeft.to(self.device)
            imgs2= imgRight.to(self.device)
            imgs3= imgBehind.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)
            # 对 force_tensor 进行归一化 #250110
            #scale_pose = [0.6, 0, 0.2, 2, 2, 1]  # 归一化比例 #1/2
            #for i in range(6):  # 对每一列进行归一化
            #    labels[:, i] = labels[:, i] / scale_pose[i]

            # print(cnt, imgs.shape)
            # print(imgs.is_cuda)
            outputs = reload_model(imgs1, imgs2,imgs3, jc)
            # print(outputs.shape, outputs[:,0:3].shape)
            l1 = vloss(outputs[:,0:3], labels[:,0:3])
            l2 = vloss(outputs[:,3:6], labels[:,3:6])
            #loss = 0.1*l1 + 10*l2
            loss = l1 + l2


            l_rx = vloss(outputs[:, 3], labels[:, 3])
            l_ry = vloss(outputs[:, 4], labels[:, 4])
            l_rz = vloss(outputs[:, 5], labels[:, 5])
            #loss = 1 * l1 + 0.5 * (l_rx + l_ry )+ 1*l_rz
            
            # print(0.1*l1,l2*3)
            # loss = vloss(outputs, labels)
            running_loss += loss.item()
            cnt = cnt + 1
        # print("var loss:", running_loss/len(var_data))
        return running_loss/cnt
    
    @torch.no_grad()
    def varGeo(self, varData, model = None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=varData, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        if model==None:
            # reload_model = PNet(self.pre_model_pth).to(self.device)
            reload_model = PNet(self.saved_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        vloss = torch.nn.MSELoss()
        # 
        reload_model.pose_MPL.eval()
        # reload_model.eval()
            
        total_transation_err = 0.0
        total_rotation_err = 0.0
        
        for i, datax in enumerate(traing_data):
            imgLeft= datax[0]
            imgRight= datax[1]
            imgBehind = datax[2]  # 新增的处理 behind 数据 #LiS240826 behind
            joint_config = datax[3]
            objectPose = datax[4]

            imgs1= imgLeft.to(self.device)
            imgs2= imgRight.to(self.device)
            imgs3 = imgBehind.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            # print(cnt, imgs.shape)
            # print(imgs.is_cuda)
            outputs = reload_model(imgs1, imgs2, imgs3, jc)

            # print(outputs[])
            # print(outputs.detach().cpu().numpy())
            transation_err, rotation_err = self.measureDistance(outputs, labels)
            total_transation_err += transation_err
            total_rotation_err += rotation_err

        # print("var loss:", running_loss/len(traing_data))
        return total_transation_err/len(traing_data), total_rotation_err/len(traing_data)*180.0/math.pi
    
    # Sattler T, Benchmarking 6dof outdoor visual localization in changing conditions, CVPR, 2018
    def measureDistance(self, predict_pose, true_pose):
        predict_pose = predict_pose.detach().cpu().numpy()
        true_pose = true_pose.detach().cpu().numpy()
        if len(predict_pose) != len(true_pose):
            raise Exception("predice and label size are not equal!")

        transitionError = []
        rotationError = []
        cnt = 0
        for i in range(len(predict_pose)):
            temp_trans = np.linalg.norm(predict_pose[i,0:3]-true_pose[i,0:3])

            temp_transfer1 = spR.from_rotvec(predict_pose[i,3:6])
            temp_transfer2 = spR.from_rotvec(true_pose[i,3:6])
            predict_matrix = temp_transfer1.as_matrix()
            true_matrix = temp_transfer2.as_matrix()

            err_matrix = np.dot(np.linalg.inv(true_matrix), predict_matrix)
            angle = np.abs(np.arccos((np.trace(err_matrix)-1)/2))

            # if temp_trans>100 or angle*180/3.14>50:
            #     print("error:",temp_trans,angle)
            #     cnt = cnt +1
            #     continue

            # if abs(predict_pose[i,0]) >250 or abs(predict_pose[i,1]) >520 or abs(predict_pose[i,2]) >200:
            #     print("error:",temp_trans,angle)
            #     cnt = cnt +1
            #     continue


            transitionError.append(temp_trans)
            rotationError.append(angle)
        # print("=============cnt===========", cnt)
        return np.mean(transitionError), np.mean(rotationError)
    
# test data 
    @torch.no_grad()
    def varGeoList(self, varData, model = None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=varData, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        if model==None:
            # reload_model = PNet(self.pre_model_pth).to(self.device)
            reload_model = PNet(self.saved_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        reload_model.pose_MPL.eval()
        # reload_model.eval()
            
        total_transation_err = np.array([])
        total_rotation_err = np.array([])
        
        for i, datax in enumerate(traing_data):
            imgLeft= datax[0]
            imgRight= datax[1]
            imgBehind = datax[2]  # 新增的处理 behind 数据 #LiS240826 behind
            joint_config = datax[3]
            objectPose = datax[4]

            imgs1= imgLeft.to(self.device)
            imgs2= imgRight.to(self.device)
            imgs3=imgBehind.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            # print(cnt, imgs.shape)
            # print(imgs.is_cuda)
            outputs = reload_model(imgs1, imgs2, imgs3, jc)

            # print(outputs[])
            # print(outputs.detach().cpu().numpy())
            transation_err, rotation_err = self.measureDistanceList(outputs, labels)
            total_transation_err = np.hstack((total_transation_err,transation_err))
            total_rotation_err = np.hstack((total_rotation_err,rotation_err))

        # print("var loss:", running_loss/len(traing_data))
        return total_transation_err, total_rotation_err
    
# test data,each components
    @torch.no_grad()
    def varGeoListDimension(self, varData, model = None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=varData, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        if model==None:
            # reload_model = PNet(self.pre_model_pth).to(self.device)
            reload_model = PNet(self.saved_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            reload_model = model

        reload_model.pose_MPL.eval()
        # reload_model.eval()
            
        total_pre = 0
        total_gt = 0
        
        for i, datax in enumerate(traing_data):
            imgLeft= datax[0]
            imgRight= datax[1]
            imgBehind = datax[2]  # 新增的处理 behind 数据 #LiS240826 behind
            joint_config = datax[3]
            objectPose = datax[4]

            imgs1= imgLeft.to(self.device)
            imgs2= imgRight.to(self.device)
            imgs3= imgBehind.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            # print(cnt, imgs.shape)
            # print(imgs.is_cuda)
            outputs = reload_model(imgs1, imgs2, imgs3, jc)

            predict_pose = outputs.detach().cpu().numpy()
            true_pose = labels.detach().cpu().numpy()

            if type(total_gt)==int:
                total_pre=predict_pose
                total_gt=true_pose
            else:
                total_pre = np.vstack((total_pre,predict_pose))
                total_gt = np.vstack((total_gt,true_pose))

        return total_pre, total_gt
    
    def measureDistanceList(self, predict_pose, true_pose):
        predict_pose = predict_pose.detach().cpu().numpy()
        true_pose = true_pose.detach().cpu().numpy()
        if len(predict_pose) != len(true_pose):
            raise Exception("predice and label size are not equal!")

        transitionError = []
        rotationError = []
        cnt = 0
        for i in range(len(predict_pose)):
            temp_trans = np.linalg.norm(predict_pose[i,0:3]-true_pose[i,0:3])

            # 将度数转换为弧度
            outputs_rad = np.deg2rad(predict_pose[i, 3:6])#240910
            labels_rad = np.deg2rad(true_pose[i, 3:6])#240910
            # 将弧度旋转向量转换为旋转矩阵
            temp_transfer1 = spR.from_rotvec(outputs_rad)
            temp_transfer2 = spR.from_rotvec(labels_rad)
            #temp_transfer1 = spR.from_rotvec(predict_pose[i,3:6])
            #temp_transfer2 = spR.from_rotvec(true_pose[i,3:6])
            predict_matrix = temp_transfer1.as_matrix()
            true_matrix = temp_transfer2.as_matrix()

            err_matrix = np.dot(np.linalg.inv(true_matrix), predict_matrix)
            angle = np.abs(np.arccos((np.trace(err_matrix)-1)/2))

            transitionError.append(temp_trans)
            rotationError.append(angle)
        # print("=============cnt===========", cnt)
        return np.array(transitionError), np.array(rotationError)



def loss_geo(output, targte):
    if len(output) != len(targte):
        raise Exception("predice and label size are not equal!")
    tx = torch.norm(output[:,0:3]-targte[:,0:3], dim=1)
    tx = torch.mean(tx)
    # rotation error
    total_alpha = 0
    for i in range(len(output)):
        output_matrix = rv2matrix(output[i,3:6])
        targte_matrix = rv2matrix(targte[i,3:6])
        
        temp_alpha = rotationError(output_matrix, targte_matrix)
        total_alpha += temp_alpha
    tr = total_alpha/len(output)
    
    return tx+ tr
                                                                                                                                                           
# Rodrigues' formula
# rotation vector to rotation matirx, Tomasi, Carlo. "Vector representation of rotations." Computer Science 527 (2013): 2-4.
def rv2matrix(rv, device = torch.device("cuda:0")):
    rv = rv.to(torch.device("cpu"))
    # for single tensor                                                     
    theta = torch.norm(rv)
    if theta == 0:
        RodriguesMatrix = torch.eye(3)
    else:
        u = rv/theta
        ux = torch.Tensor([[0, -u[2],u[1]],[u[2], 0, -u[0]],[-u[1], u[0], 0]])
        RodriguesMatrix = torch.cos(theta)*torch.eye(3) + (1 - torch.cos(theta))* (u.reshape(1,-1) * u.reshape(-1,1)) + ux *torch.sin(theta)
    return RodriguesMatrix.to(device)


# Sattler T, Benchmarking 6dof outdoor visual localization in changing conditions, CVPR, 2018
# angle betweent rotations, radian
def rotationError(estimate_matrix, target_matrix):
    err_matrix = torch.inverse(target_matrix) @ estimate_matrix
    alpha = torch.abs(torch.acos((torch.trace(err_matrix)-1)/2))
    return alpha


    
            




import time
if __name__ == "__main__":


    test = TrainRegression()


