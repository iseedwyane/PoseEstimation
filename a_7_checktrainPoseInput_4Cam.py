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
#from PoseNet import PNet, EncordFun
#from LoadData import ReconstDataset
from torchvision import transforms, utils
from scipy.spatial.transform import Rotation as spR
import cv2
import copy
import random
import os
import matplotlib.pyplot as plt
import csv

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
    
    return tx + tr

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

class TrainRegression(object):
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.momentum = 0.3
        self.epoch = 200
        self.pre_model_pth = './weights/reconstruction_weight_LS_64_3bjects_240914.pth'#1/2
        #self.pre_model_pth = "./weights/reconstruction_weight_PEACH_241226.pth"#1/2
        self.saved_pth = "./weights/pose_weight.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.recons_loss = torch.nn.MSELoss()
        self.data_folder = "./IMG_DATA"
        self.data_folder_var = "./IMG_DATA"
        print("============data_folder_var_invaild",self.data_folder_var,"============")
        self.width = 320
        self.height = 320

    def augment(self, data):
        probability = random.uniform(0,1)
        if probability < 0.4:
            return data
        elif probability >= 0.4 and probability < 0.8:
            jitter = transforms.ColorJitter(brightness=.4, contrast=.3, saturation=0.2)
            trans_data = [jitter(data[i]) for i in range(len(data))]
            return torch.stack(trans_data)
        elif probability >= 0.8:
            blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.5))
            trans_data = [blurrer(data[i]) for i in range(len(data))]
            return torch.stack(trans_data)

    def train(self):
        dataset_image = ReconstDataset(data_path=self.data_folder, scaled_width=self.width, scale_height=self.height)
        print("sample numbers: ", len(dataset_image))
        print("sample dataset_image[3]'s len: ", len(dataset_image[3]))

        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=16, drop_last=False, pin_memory=False, collate_fn=None)
        
        pose_model = PNet(self.pre_model_pth).to(self.device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, pose_model.parameters()), lr=self.learning_rate)
        vloss = torch.nn.MSELoss()

        pre_loss = 10e10
        epoch_train_loss = []
        epoch_var_loss = []
        
        for epoch in range(self.epoch):
            pose_model.pose_MPL.train()
            running_loss = 0.0
            cnt = 0
            for i, datax in enumerate(traing_data):
                imgLeft = datax[0]
                imgRight = datax[1]
                joint_config = datax[2]
                objectPose = datax[3]

                imgs1 = imgLeft.to(self.device)
                imgs2 = imgRight.to(self.device)
                jc = joint_config.to(self.device)
                labels = objectPose.to(self.device)
                labels = labels.squeeze(1)

                optimizer.zero_grad()
                outputs = pose_model(imgs1, imgs2, jc)
                
                l1 = vloss(outputs[:,0:3], labels[:,0:3])
                l2 = vloss(outputs[:,3:6], labels[:,3:6])
                loss = 0.1*l1 + 10*l2
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                cnt += 1
                
            print('epoch: %d, train loss: %.6f, pre_loss: %.6f' % (epoch, running_loss/cnt, pre_loss))
            if running_loss/cnt < pre_loss:
                pre_loss = running_loss/cnt
                print("======save weights======", "epoch:",epoch)
                torch.save(pose_model.state_dict(), self.saved_pth)

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
        running_loss = 0.0
        cnt = 0
        for i, datax in enumerate(var_data):
            imgLeft = datax[0]
            imgRight = datax[1]
            joint_config = datax[2]
            objectPose = datax[3]

            imgs1 = imgLeft.to(self.device)
            imgs2 = imgRight.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            outputs = reload_model(imgs1, imgs2, jc)
            
            l1 = vloss(outputs[:,0:3], labels[:,0:3])
            l2 = vloss(outputs[:,3:6], labels[:,3:6])
            loss = 0.1*l1 + 10*l2
            
            running_loss += loss.item()
            cnt = cnt + 1

        return running_loss/cnt

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
            joint_config = datax[2]
            objectPose = datax[3]

            imgs1 = imgLeft.to(self.device)
            imgs2 = imgRight.to(self.device)
            jc = joint_config.to(self.device)
            labels = objectPose.to(self.device)
            labels = labels.squeeze(1)

            outputs = reload_model(imgs1, imgs2, jc)

            transation_err, rotation_err = self.measureDistance(outputs, labels)
            total_transation_err += transation_err
            total_rotation_err += rotation_err

        return total_transation_err/len(traing_data), total_rotation_err/len(traing_data) * 180.0 / math.pi
    
    def measureDistance(self, predict_pose, true_pose):
        predict_pose = predict_pose.detach().cpu().numpy()
        true_pose = true_pose.detach().cpu().numpy()
        if len(predict_pose) != len(true_pose):
            raise Exception("predice and label size are not equal!")

        transitionError = []
        rotationError = []
        for i in range(len(predict_pose)):
            temp_trans = np.linalg.norm(predict_pose[i,0:3] - true_pose[i,0:3])

            temp_transfer1 = spR.from_rotvec(predict_pose[i,3:6])
            temp_transfer2 = spR.from_rotvec(true_pose[i,3:6])
            predict_matrix = temp_transfer1.as_matrix()
            true_matrix = temp_transfer2.as_matrix()

            err_matrix = np.dot(np.linalg.inv(true_matrix), predict_matrix)
            angle = np.abs(np.arccos((np.trace(err_matrix) - 1) / 2))

            transitionError.append(temp_trans)
            rotationError.append(angle)

        return np.mean(transitionError), np.mean(rotationError)

def plot_objectPose_distribution(data_folder, num_bins=50, folder_name=""):
    # 加载数据
    dataset_image = ReconstDataset(data_path=data_folder, scaled_width=320, scale_height=320)
    print(f"样本数量 ({folder_name}): ", len(dataset_image))

    # 初始化数组存储所有的 objectPose 数据
    objectPose_data = []

    # 遍历数据集提取 objectPose 数据
    for i in range(len(dataset_image)):
        _, _, _, _, objectPose = dataset_image[i]
        objectPose_data.append(objectPose.numpy().flatten())

    # 将列表转换为 numpy 数组
    objectPose_data = np.array(objectPose_data)

    # 分离每个 objectPose 分量到单独的数组
    x_data = objectPose_data[:, 0]
    y_data = objectPose_data[:, 1]
    z_data = objectPose_data[:, 2]
    Rx_data = objectPose_data[:, 3]
    Ry_data = objectPose_data[:, 4]
    Rz_data = objectPose_data[:, 5]

    # 保存到CSV文件
    csv_file_path = os.path.join("./IMG_DATA_LS/", f"{folder_name}_objectPose_data.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'z', 'Rx', 'Ry', 'Rz'])
        for i in range(len(x_data)):
            writer.writerow([x_data[i], y_data[i], z_data[i], Rx_data[i], Ry_data[i], Rz_data[i]])

    print(f"数据已保存到 {csv_file_path}")

    # 绘制每个 objectPose 分量的直方图
    fig, axes = plt.subplots(1, 6, figsize=(18, 5))
    fig.suptitle(f'{folder_name} - objectPose Distribution', fontsize=16)

    labels = ['x', 'y', 'z', 'Rx', 'Ry', 'Rz']

    for i in range(6):
        plt.subplot(1, 6, i+1)
        plt.hist(objectPose_data[:, i], bins=num_bins, alpha=0.75)
        plt.xlabel(labels[i])
        plt.ylabel('Frequency')

    save_path = os.path.join("./IMG_DATA_LS/", f"{folder_name}_distribution.png")
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test = TrainRegression()

    data_path = ["IMG_DATA_MASTERBALL_0907","IMG_DATA_PEACH_0912","IMG_DATA_COVER_0912"]#2/2
    #data_path = ["IMG_DATA_PEACH_241226"]#2/2
    for i in range(len(data_path)):
        folder_name = data_path[i]
        test.data_folder = "./IMG_DATA_LS/" + data_path[i] + "/train.txt"
        test.data_folder_var = "./IMG_DATA_LS/" + data_path[i] + "/all.txt"        
        print("=================",data_path[i],"===================")
   
        # 调用 plot_objectPose_distribution 方法绘制直方图并保存数据到CSV
        plot_objectPose_distribution(test.data_folder, folder_name=folder_name)
