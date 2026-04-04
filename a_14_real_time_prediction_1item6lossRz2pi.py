import torch
import cv2
import numpy as np
from PoseNet_4Cam import PNet
from scipy.spatial.transform import Rotation as spR
import os
import glob
import csv
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random

class RealTimePrediction:
    def __init__(self):
        self.pre_model_pth = "./weights/reconstruction_weight_LS_64_3bjects_240912.pth"
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

    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        img = img.astype(np.float32) / 255.0  # 归一化
        img = cv2.resize(img, (self.width, self.height))  # 调整大小
        return img

class ReadAruCo:
    def __init__(self):
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.016

    def readPose(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)

        if ids is None or len(ids) != 4:
            return False, False

        indexs = ids.argsort(axis=0).reshape(-1)
        order_ids = ids[indexs]
        orders_corners = np.array(corners)[indexs]
        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(orders_corners, self.mark_size, self.camera_matrix, self.camera_dist)

        for i in range(len(ids)):
            color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)

        return np.hstack((order_ids, tvec.reshape(-1, 3), rvec.reshape(-1, 3))), color_image_result

class RawDataTransfer(object):
    def __init__(self, init_path="") -> None:
        # 读取初始参数姿态
        self.path = init_path

        # 初始姿态
        init_gripper_data, init_object_data, init_j1_data, init_j2_data = self.readInitPose(self.path)
        self.init_gripper_matrix = self.pose2matrix(init_gripper_data)
        self.init_object_matrix = self.pose2matrix(init_object_data)
        self.init_j1_matrix = self.pose2matrix(init_j1_data)
        self.init_j2_matrix = self.pose2matrix(init_j2_data)

    def pose2matrix(self, pose):
        # 6d 位姿转换为4×4矩阵
        transation_vector = pose[0:3]
        rotation_vector = pose[3:6]
        rr = spR.from_rotvec(rotation_vector)
        rotation_matrix = rr.as_matrix()
        matrix = np.zeros((4, 4))
        matrix[3, 3] = 1
        matrix[0:3, 0:3] = rotation_matrix
        matrix[0:3, 3] = transation_vector
        return matrix

    def readInitPose(self, init_folder=""):
        # 平均姿态作为初始姿态
        file_list = glob.glob(os.path.join(init_folder, "**.txt"))

        gripper_list = []
        object_list = []
        joint1 = []
        joint2 = []

        for i in range(len(file_list)):
            temp = np.loadtxt(file_list[i])
            joint1.append(temp[0, 1:7])
            object_list.append(temp[1, 1:7])
            gripper_list.append(temp[2, 1:7])
            joint2.append(temp[3, 1:7])

        gripper_list = np.array(gripper_list)
        object_list = np.array(object_list)
        joint1 = np.array(joint1)
        joint2 = np.array(joint2)

        mg = spR.from_rotvec(gripper_list[:, 3:6])
        gripper_r_init = mg.mean().as_rotvec()
        gripper_t_init = np.mean(gripper_list[:, 0:3], axis=0)

        mo = spR.from_rotvec(object_list[:, 3:6])
        object_r_init = mo.mean().as_rotvec()
        object_t_init = np.mean(object_list[:, 0:3], axis=0)

        mj1 = spR.from_rotvec(joint1[:, 3:6])
        mj1_r_init = mj1.mean().as_rotvec()
        mj1_t_init = np.mean(joint1[:, 0:3], axis=0)

        mj2 = spR.from_rotvec(joint2[:, 3:6])
        mj2_r_init = mj2.mean().as_rotvec()
        mj2_t_init = np.mean(joint2[:, 0:3], axis=0)

        return np.hstack((gripper_t_init, gripper_r_init)), np.hstack((object_t_init, object_r_init)), \
               np.hstack((mj1_t_init, mj1_r_init)), np.hstack((mj2_t_init, mj2_r_init))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型加载
    saved_pth = ["./weights/pose_weight_masterball_0912.pth", "./weights/pose_weight_peach_0912.pth", "./weights/pose_weight_cover_0912.pth"]
    weight_path = saved_pth[0]
    pre_model_pth = "./weights/reconstruction_weight_LS_64_3bjects_240912.pth"
    pose_model = PNet(pre_model_pth).to(device)
    pose_model.load_state_dict(torch.load(weight_path))
    pose_model.eval()

    prediction = RealTimePrediction()
    MarkerDetector = ReadAruCo()

    # 随机选择文件编号
    file_range_start = 1
    file_range_end = 5000
    random_file_number = random.randint(file_range_start, file_range_end)
    random_file_name = f'{random_file_number:08d}'

    print("random_file_name:", random_file_name)
    # 构造随机文件路径
    img_left_path = f'./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/IMG_LEFT/{random_file_name}.jpg'
    img_right_path = f'./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/IMG_RIGHT/{random_file_name}.jpg'
    img_behind_path = f'./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/IMG_BEHIND/{random_file_name}.jpg'
    img_outside_path = f'./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/IMG_OUTSIDE/{random_file_name}.jpg'

    # 读取随机选择的图像
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    img_behind = cv2.imread(img_behind_path)
    img_outside = cv2.imread(img_outside_path)

    if img_left is None or img_right is None or img_behind is None or img_outside is None:
        print(f"Error: Could not read one or more images for file number {random_file_name}.")
    else:
        # 图像预处理
        img1_data = prediction.preprocess_image(img_left)
        img2_data = prediction.preprocess_image(img_right)
        img3_data = prediction.preprocess_image(img_behind)

        # ArUco标记检测
        result_pose, result_img = MarkerDetector.readPose(img_outside)
        if result_pose is not False:
            pose_11 = None
            pose_14 = None
            for pose in result_pose:
                if pose[0] == 11:
                    pose_11 = pose[1:4]
                elif pose[0] == 14:
                    pose_14 = pose[1:4]
                elif pose[0] == 13:
                    gripper_pose = pose[1:7]
                elif pose[0] == 12:
                    object_pose = pose[1:7]
                else:
                    print("result_pose met error")

            # 计算物体在手中的位置和初始物体在手中的位置
            a = RawDataTransfer("./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/Pose_Init")
            gripper_matrix = a.pose2matrix(gripper_pose)
            object_matrix = a.pose2matrix(object_pose)
            objectInHand = np.matmul(np.linalg.inv(gripper_matrix), object_matrix)
            initObjectInHand = np.matmul(np.linalg.inv(a.init_gripper_matrix), a.init_object_matrix)

            transfer_matrix = np.matmul(np.linalg.inv(initObjectInHand), objectInHand)
            rr = spR.from_matrix(transfer_matrix[0:3, 0:3])
            rotation_vect = rr.as_rotvec()
            GroundTruthPOSE = np.hstack((transfer_matrix[0:3, 3] * 1000, rotation_vect))

            if pose_11 is not None and pose_14 is not None:
                gripper_config = np.linalg.norm(pose_11 - pose_14) * 1000
            else:
                gripper_config = 0.1030280649243982 * 1000
        else:
            gripper_config = 0.1030280649243982 * 1000
            print("No ID 12, gripper_config met error")

        print("gripper_config:", gripper_config)

        imgLeft = torch.Tensor(np.array([img1_data])).to(device)
        imgRight = torch.Tensor(np.array([img2_data])).to(device)
        imgBehind = torch.Tensor(np.array([img3_data])).to(device)
        joint_config = torch.Tensor(np.array([gripper_config])).to(device)

        imgLeft = imgLeft.unsqueeze(0)
        imgRight = imgRight.unsqueeze(0)
        imgBehind = imgBehind.unsqueeze(0)
        joint_config = joint_config.unsqueeze(0)

        # 模型推理
        outputs = pose_model(imgLeft, imgRight, imgBehind, joint_config)

        # 将输出和真实值转换为张量
        GroundTruthPOSE_tensor = torch.tensor(GroundTruthPOSE, dtype=torch.float32).unsqueeze(0).to(device)

        # 分别计算每个维度的损失
        criterion = torch.nn.L1Loss()
        losses = {}
        for i, axis in enumerate(['x(mm)', 'y(mm)', 'z(mm)', 'rx(degree)', 'ry(degree)', 'rz(degree)']):
            loss = criterion(outputs[:, i], GroundTruthPOSE_tensor[:, i])
            if axis in ['rx(degree)', 'ry(degree)', 'rz(degree)']:
                # 将弧度转换为度后计算损失
                loss = torch.sqrt(loss)
            losses[axis] = loss.item()

        print("Predicted Pose:", outputs)
        print("Ground Truth Pose:", GroundTruthPOSE)
        print("Losses:")
        for axis, loss in losses.items():
            print(f"{axis} Loss: {loss:.6f}")

        # 计算总损失
        loss = criterion(outputs, GroundTruthPOSE_tensor)
        print("Total Loss:", loss.item())
