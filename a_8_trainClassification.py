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
        self.batch_size = 256
        self.learning_rate = 0.001
        self.momentum = 0.3
        self.epoch = 100
        self.pre_model_pth = "./weights/reconstruction_weight_LS_64.pth"
        self.saved_pth = "./weights/cls_weight_LS.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        # loss
        self.recons_loss =torch.nn.MSELoss()
        self.data_folder = "./IMG_DATA_LS"
        self.data_folder_var = "./IMG_DATA_LS"
        self.width = 320
        self.height = 320

    
    def train(self):
        # load data
        dataset_image = ClassificationDataset(data_path=self.data_folder, data_mode = "train", scaled_width = self.width, scale_height= self.height)
        print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=False, collate_fn=None)
        
        cls_model = CNet(self.pre_model_pth).to(self.device)
        # optimizer
        # optimizer = optim.SGD(pose_model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        # optimizer = optim.Adam(cls_model.parameters(), lr=self.learning_rate)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, cls_model.parameters()), lr=self.learning_rate)
        vloss = torch.nn.CrossEntropyLoss()

        pre_loss = 10e10
        epoch_train_loss = []
        epoch_var_loss = []
        for epoch in range(self.epoch):
            # training
            cls_model.cls_MPL.train()
            running_loss = 0.0
            cnt = 0
            for i, datax in enumerate(traing_data):
                imgLeft= datax[0]
                imgRight= datax[1]
                joint_config = datax[2]
                labels = datax[3]
                

                imgs1= imgLeft.to(self.device)
                imgs2= imgRight.to(self.device)
                jc = joint_config.to(self.device)
                labels = labels.to(self.device)
                labels = labels.squeeze(1)
                # print(labels.shape)
                
                optimizer.zero_grad()
                outputs = cls_model(imgs1, imgs2, jc)
                # print("labels:",labels, np.argmax(outputs.detach().cpu().numpy(), axis=1))
                loss = vloss(outputs, labels)
                # loss = outputs
                # print("loss:",loss, loss.is_leaf, loss.grad)
                

                # print('batch size: ', i, imgs1.shape, len(labels))
                # print('labels: ', labels.shape, labels.get_device())
                # print('outputs: ', outputs.shape, outputs.get_device())
                # print("epoch %d, batch %d," % (epoch, i), 'loss: ', loss.item())
                # optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=pose_model.parameters(),max_norm=10)
                optimizer.step()
                running_loss += loss.item()
                cnt += 1
                # if i % 20 == 19:
                #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/20))
                #     running_loss = 0.0

            # save loss
            epoch_train_loss.append(running_loss/cnt)
            
            print('epoch: %d, train loss: %.6f, pre_loss: %.6f' % (epoch, running_loss/cnt, pre_loss))
            # save weight
            if epoch == 0:
                torch.save(cls_model.state_dict(), self.saved_pth)
                pre_loss = running_loss/cnt
            # elif epoch%10==0:
            else:
                # var_loss = self.var(self.data_folder_var ,model=pose_model)
                var_loss = self.var(datamode="train", model=cls_model)
                print("epoch:", epoch, ", var_loss:", var_loss)
                if var_loss<pre_loss:
                    pre_loss = var_loss
                    print("======save weights======", "epoch:",epoch)
                    torch.save(cls_model.state_dict(), self.saved_pth)

        # # save weight
        # np.savetxt('./weights/pw_trainLoss.txt', epoch_train_loss)
        # np.savetxt('./weights/pw_varLoss.txt', epoch_var_loss)

    @torch.no_grad()
    def var(self, datamode='var' , model = None, trained_weights = "./weights/cls_weight.pth"):
        # 载入测试集
        dataset_image = ClassificationDataset(data_path=self.data_folder, data_mode = datamode, scaled_width = self.width, scale_height= self.height)
        # print("sample numbers: ", len(dataset_image))
        var_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=False, collate_fn=None)
        # 载入模型
        if model==None:
            reload_model = CNet(self.pre_model_pth).to(self.device)
            # if self.device.type != 'cpu':
            #     reload_model = torch.nn.DataParallel(reload_model, device_ids=[0,1])
            reload_model.load_state_dict(torch.load(trained_weights))
        else:
            # reload_model = copy.deepcopy(model)
            reload_model = model


        reload_model.cls_MPL.eval()

        # # print(reload_model.pose_MPL)
        # for child in reload_model.pose_MPL.children():
        #     if type(child)==torch.nn.BatchNorm1d:
        #         child.track_running_stats = False
        # # print(reload_model.pose_MPL)
        

        vloss = torch.nn.CrossEntropyLoss()
        # 
        running_loss = 0.0
        cnt = 0
        for i, datax in enumerate(var_data):
            imgLeft= datax[0]
            imgRight= datax[1]
            joint_config = datax[2]
            labels = datax[3]

            imgs1= imgLeft.to(self.device)
            imgs2= imgRight.to(self.device)
            jc = joint_config.to(self.device)
            labels = labels.to(self.device)
            labels = labels.squeeze(1)

            # print(cnt, imgs.shape)
            # print(imgs.is_cuda)
            outputs = reload_model(imgs1, imgs2, jc)
            # print(outputs.shape, outputs[:,0:3].shape)

            loss = vloss(outputs, labels)
            running_loss += loss.item()
            cnt = cnt + 1
        # print("var loss:", running_loss/len(var_data))
        return running_loss/cnt

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
    # test = TrainRegression()
    # test.train()
    # x1 = test.var(test.data_folder, trained_weights = "./weights/pose_weight.pth")
    # x2 = test.varGeo(test.data_folder, trained_weights = "./weights/pose_weight.pth")
    # print("train:",x1, x2)
    # x1 = test.var(test.data_folder_var, trained_weights = "./weights/pose_weight.pth")
    # x2 = test.varGeo(test.data_folder_var, trained_weights = "./weights/pose_weight.pth")
    # print("var:",x1, x2)

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



    # training
    test = TrainClassification()
    test.train()

    


