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
import cv2
import copy

# torch.backends.cudnn.enabled=False
class TrainRegression(object):
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 0.01
        self.momentum = 0.3
        self.epoch = 3000
        self.pre_model_pth = "./weights/reconstruction_weight_64.pth"
        self.saved_pth = "./weights/pose_weight.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        # loss
        self.recons_loss =torch.nn.MSELoss()
        self.data_folder = "./IMG_DATA/IMG_DATA_tube1/train.txt"
        self.data_folder_var = "./IMG_DATA/IMG_DATA_tube1/train.txt"
        self.width = 320
        self.height = 320

        # load dataset
        dataset_image = ReconstDataset(data_path=self.data_folder, scaled_width = self.width, scale_height= self.height)
        print("train sample numbers: ", len(dataset_image))
        self.traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)

        dataset_image_var = ReconstDataset(data_path=self.data_folder_var, scaled_width = self.width, scale_height= self.height)
        print("var sample numbers: ", len(dataset_image_var))
        self.var_data = torch.utils.data.DataLoader(dataset=dataset_image_var, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)

        self.model = PNet(self.pre_model_pth).to(self.device)




    def train(self, traing_data=0):
        traing_data = self.traing_data
        pose_model = self.model
                                                                                                            
        # optimizer
        # optimizer = optim.SGD(pose_model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        optimizer = optim.Adam(pose_model.parameters(), lr=self.learning_rate)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, pose_model.parameters()), lr=self.learning_rate)
        vloss = torch.nn.MSELoss()

        pre_loss = 10e10
        for epoch in range(self.epoch):
            # training
            pose_model.pose_MPL.train()
            running_loss = 0.0
            cnt = 0

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

                optimizer.zero_grad()

                outputs = pose_model(imgs1, imgs2, jc)
                # loss = loss_geo(outputs, labels)
                loss = vloss(outputs, labels)
                
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=pose_model.parameters(),max_norm=10)
                optimizer.step()
                
                # statistics
                # running_loss += loss.item() * labels.size(0)
                running_loss += loss.item()
                cnt += 1

            #  show tunning_var
            # print("train running param:",pose_model.encoder_model._encoder.enc_norm0.running_mean, pose_model.encoder_model._encoder.enc_norm0.running_var)
            # save loss
            epoch_train_loss = running_loss/cnt
            
            print('epoch: %d, train loss: %.6f, pre_loss: %.6f' % (epoch, epoch_train_loss, pre_loss))
            # save weight
            if epoch == 0:
                torch.save(pose_model.state_dict(), self.saved_pth)
                pre_loss = epoch_train_loss
            else:
                # var_loss = self.var(model=pose_model)
                # print("epoch:", epoch, ", var_loss:", var_loss)
                if epoch_train_loss < pre_loss:
                    pre_loss = epoch_train_loss
                    torch.save(pose_model.state_dict(), self.saved_pth)

    @torch.no_grad()
    def var(self, model=None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        var_data = self.var_data
        # 载入模型
        # reload_model = model

        # 载入模型
        if model==None:
            reload_model = PNet(self.pre_model_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            # reload_model = copy.deepcopy(self.model)
            reload_model = model

        reload_model.eval()
        # print("var running param:",reload_model.encoder_model._encoder.enc_norm0.running_mean, reload_model.encoder_model._encoder.enc_norm0.running_var)
        
        vloss = torch.nn.MSELoss()
        # 
        running_loss = 0.0
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
            loss = vloss(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            epoch_train_loss = running_loss/len(var_data.dataset)
        # print("var loss:", running_loss/len(var_data))
        return epoch_train_loss
    
    @torch.no_grad()
    def varGeo(self, model = None, trained_weights = "./weights/pose_weight.pth"):
        # 载入测试集
        dataset_image = ReconstDataset(data_path=self.data_folder_var, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
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


    
            





if __name__ == "__main__":
    test = TrainRegression()
    test.train()
    
    # x_mse = test.var(trained_weights = "./weights/pose_weight.pth")
    # x_geo = test.varGeo(trained_weights = "./weights/pose_weight.pth")
    # print(x_mse, x_geo)

    # reload_model = PNet("./weights/reconstruction_weight_1024.pth")
    # reload_model.load_state_dict(torch.load("./weights/pose_weight.pth"))
    # print(reload_model.encoder_model._encoder.enc_conv0.bias)


    # a =torch.load("./weights/pose_weight.pth")
    # print(a.keys())
    # print(a['encoder_model._encoder.enc_norm0.weight'], a['encoder_model._encoder.enc_norm0.bias'])
    # print(a['encoder_model._encoder.enc_norm0.running_mean'], a['encoder_model._encoder.enc_norm0.running_var'])
    # print(a['encoder_model._encoder.enc_norm0.num_batches_tracked'])

    # b = torch.load("./weights/reconstruction_weight_128.pth")
    # # print(b['_encoder.enc_conv1.weight'])
    # # c = a['encoder_model._encoder.enc_conv0.weight']-b['_encoder.enc_conv0.weight']
    # d = a['encoder_model._encoder.enc_norm0.weight']-b['_encoder.enc_norm0.weight']
    # e = a['encoder_model._encoder.enc_norm0.bias']-b['_encoder.enc_norm0.bias']
    # f = a['encoder_model._encoder.enc_norm0.running_mean']-b['_encoder.enc_norm0.running_mean']
    # g = a['encoder_model._encoder.enc_norm0.running_var']-b['_encoder.enc_norm0.running_var']
    # print("xx:", torch.where(d != 0), torch.where(e != 0), torch.where(f != 0), torch.where(g != 0))







