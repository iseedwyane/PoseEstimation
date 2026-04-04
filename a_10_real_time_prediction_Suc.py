import torch
import cv2
import numpy as np
from PoseNet import PNet
from LoadData import ReconstDataset
from scipy.spatial.transform import Rotation as spR
from torchvision import transforms
import cv2
import os
import queue
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as spR
from datetime import datetime

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
from torchvision import transforms,utils

from scipy.spatial.transform import Rotation as spR
import cv2
import copy
import random

from torch.utils.data import DataLoader, Dataset
from AutoEncoder import AE

class RealTimePrediction:
    def __init__(self):     
        self.pre_model_pth = "./weights/reconstruction_weight_PEACH_241226.pth"#3/4
        self.width = 320
        self.height = 320
        self.batch_size = 32

        self.saved_pth = "./weights/pose_weight.pth"
        self.data_folder = "./IMG_DATA"  # 任意路径，因为训练不需要此输入
        self.data_folder_var = "./IMG_DATA"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        #print("============__init__============")

    def preprocess_image(self, img):

        # 转换颜色空间和数据类型
        img1_data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        img1_data = img1_data/ 255.0  # 归一化到[0, 1]  
        img1_data = cv2.resize(img1_data, dsize=(self.width, self.height))
        height, width = img1_data.shape
        #print(f"Width of img: {width}, Height of img: {height}")      
        #return torch.Tensor(np.array([img1_data])).unsqueeze(0).to(self.device) 
        return img1_data

    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        img = img.astype(np.float32) / 255.0  # 归一化
        img = cv2.resize(img, (320, 320))  # 调整大小
        #print(f"Width of img: {width}, Height of img: {height}")
        #img = np.expand_dims(img, axis=0)  # 添加通道维度
        #img = torch.from_numpy(img)  # 转换为torch张量
        #img = img.unsqueeze(0)  # 添加批次维度
        return img        

class MultiVideoCapture:
    """ 3 cameras"""
    def __init__(self, cameraID_outside=0, cameraID_left=6, cameraID_right=8, width=640, height=360):
        #  camera for marker
        self.cap_outside = cv2.VideoCapture(cameraID_outside)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        #  camera for left finger
        self.cap_left = cv2.VideoCapture(cameraID_left)
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #  camera for left finger
        self.cap_right = cv2.VideoCapture(cameraID_right)
        self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q_outside = queue.Queue()
        self.q_left = queue.Queue()
        self.q_right = queue.Queue()

        self.flag = True

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.flag:
            ret1, frame_outside = self.cap_outside.read()
            ret2, frame_left = self.cap_left.read()
            ret3, frame_right = self.cap_right.read()
            if not (ret1 and ret2 and ret3):
                break
            if not self.q_outside.empty():
                try:
                    self.q_outside.get_nowait()   # discard previous (unprocessed) frame
                    self.q_left.get_nowait()
                    self.q_right.get_nowait()
                except queue.Empty:
                    pass
            self.q_outside.put(frame_outside)
            self.q_left.put(frame_left)
            self.q_right.put(frame_right)

    def read(self):
        return self.q_outside.get(), self.q_left.get(), self.q_right.get()

    def stop(self):
        self.flag = False
        self.cap_outside.release()
        self.cap_left.release()
        self.cap_right.release()


"""read aruco code pose"""
class readAruCo():
    # read Aruco pose from image
    def __init__(self) -> None:
        # camera 0
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.016

    def readPose(self,img):
        # return tuple(A,B): A is data, B is image
        #  A: [[marker_id, x,y,z,rx,ry,rz]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.float32)/25
        gray = cv2.filter2D(gray,-1,kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        # if ids.any()==None:
        #     return False,False
        if ids is None or len(ids) != 4:
            return False,False
        
        indexs = ids.argsort(axis=0)
        indexs = indexs.reshape(-1)

        order_ids = ids[indexs]
        corners = np.array(corners)
        orders_corners = corners[indexs]
        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(orders_corners, self.mark_size, self.camera_matrix, self.camera_dist)

        for i in range(len(ids)):
            color_image_result = cv2.aruco.drawAxis(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
        return np.hstack((order_ids, tvec.reshape(-1,3), rvec.reshape(-1,3))), color_image_result
    

    

if __name__ == "__main__":
    # 如果CUDA可用，使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CAP = MultiVideoCapture(cameraID_outside=5, cameraID_left=11, cameraID_right=9)#4/4
    #MarkerDetector = readAruCo()
    saved_pth = ["./weights/pose_weight_peach_241226.pth"]#`1/4
    #data_path = ["IMG_DATA_BASEBALL_0628", "IMG_DATA_STARW_0628", "IMG_DATA_BOTTLECOVER_0628", "IMG_DATA_PEACH_0628", "IMG_DATA_CUBE_0628"]
    weight_path = saved_pth[0]  # 选择列表中的第一个权重文件
    pre_model_pth = "./weights/reconstruction_weight_PEACH_241226.pth"#2/4
    pose_model = PNet(pre_model_pth).to(device)  # 使用第一个权重文件创建模型实例
    pose_model.load_state_dict(torch.load(weight_path))
    pose_model.eval()

    prediction = RealTimePrediction()
    while 1:
        img_outside, img_left, img_right = CAP.read()          
        #print("============CAP__init__img_left&img_right: imgs1, imgs2============")
        height, width, channels = img_left.shape
        #print(f"Width of img_left: {width}, Height of img_left: {height}")
       # 应用预处理
        img1_data = prediction.preprocess_image(img_left)
        img2_data = prediction.preprocess_image(img_right)
        #resize
        #print("============CAP__init_Get_FingerConfig: jc============")
        #result_pose,result_img = MarkerDetector.readPose(img_outside)  
        #if result_pose is not False:
        #    print(f"Dimensions of result_pose: {result_pose.shape}")
            # 输出 result_pose 的尺寸，即标记的数量和每个标记的数据维度
            # 可以进一步处理 result_pose
        #else:
        #    print("No markers detected or less than required markers detected.")
        # Process other images or perform other tasks        
        #pose_list = result_pose

        #if len(pose_list.shape)==1:
        #    print('xx:',pose_list)
        
        #gripper_config = np.linalg.norm(pose_list[0,1:4]-pose_list[3,1:4])
        gripper_config = 0.1030280649243982*1000
        #print('gripper_config:',gripper_config)
        #jc = torch.tensor([gripper_config], dtype=torch.float32).unsqueeze(0).to(device)

        #ww = torch.Tensor(np.array([img1_data])), torch.Tensor(np.array([img2_data])), torch.Tensor(np.array([gripper_config])), torch.Tensor(np.array([objectPose]))
        #ww = torch.Tensor(np.array([img1_data])), torch.Tensor(np.array([img2_data])), torch.Tensor(np.array([gripper_config]))
        #ww = ww.to(device)
        #m = DataLoader(dataset=ww, batch_size=8, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
        ##for i_batch, data  in enumerate(m):
        #    imgLeft= data[0]
        #    imgRight= data[1]
        #    joint_config = data[2]
        #    objectPose = data[3]

        #print("============Predict__init__============")
        #pose_model.pose_MPL.eval()
        #imgLeft = torch.Tensor(np.array([img1_data])).unsqueeze(0).to(device)
        #imgRight = torch.Tensor(np.array([img2_data])).unsqueeze(0).to(device)
        imgLeft = torch.Tensor(np.array([img1_data])).to(device)
        imgRight = torch.Tensor(np.array([img2_data])).to(device)
        
        #imgs= torch.cat((imgLeft, imgRight),axis = 0)
        #print("Shape of concatenated tensor:", imgs.shape)
        ##reconstruction_model = AE(1, 1).to(device)
        #outputs = reconstruction_model(imgs)
        #print("Output from the model shape:", outputs.shape)

        joint_config = torch.Tensor(np.array([gripper_config])).to(device)
        #print("Input to the model imgLeft shape:", imgLeft.shape)
        #print("Input to the model imgRight shape:", imgRight.shape)
        #print("joint_config shape:", joint_config.shape)

        imgLeft = imgLeft.unsqueeze(0)
        imgRight = imgRight.unsqueeze(0)
        joint_config = joint_config.unsqueeze(0)
        #print("test shape:",imgLeft.shape,imgRight.shape, joint_config.shape)
        outputs = pose_model(imgLeft, imgRight, joint_config)   
        print("Predicted Pose:", outputs)
        #print("Output from the model shape:", outputs.shape)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break