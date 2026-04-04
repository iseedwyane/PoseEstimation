# MIT License.
# Copyright (c) 2022 by BioicDL. All rights reserved.
# Created by LiuXb on 2022/5/11
# -*- coding:utf-8 -*-

"""
@Modified: 
@Description: Display Aruco code detection from a single camera
"""
import cv2
import queue
import threading
import numpy as np
from scipy.spatial.transform import Rotation as spR
import glob
import csv
import os
import matplotlib.pyplot as plt

class SingleVideoCapture:
    """ Single camera """
    def __init__(self, cameraID=4, width=1920, height=1080):
        # Camera for marker
        self.cap = cv2.VideoCapture(cameraID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q = queue.Queue()
        self.flag = True

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.flag:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.flag = False
        self.cap.release()


"""Read Aruco code pose"""
class ReadAruCo:
    # Read Aruco pose from image
    def __init__(self) -> None:

        # 加载相机参数
        #matrix = np.load('camera_params_outside1920x1080LiS.npz')
        #self.camera_matrix = matrix['mtx']
        #self.camera_dist = matrix['dist']
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])

        
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.014

    def readPose(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        if ids is None:
            return img

        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)
        pose_data = []
        for i in range(len(ids)):
            if ids[i] in [11, 12, 13, 14]:
                pose = np.hstack((ids[i], tvec[i].flatten(), rvec[i].flatten()))
                pose_data.append(pose)
            color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
            color_image_result = cv2.aruco.drawDetectedMarkers(color_image_result, corners, ids)
        
        
        return color_image_result, pose_data


class RawDataTransfer:
    def __init__(self, init_path=""):
        self.path = init_path
        init_gripper_data, init_object_data, init_j1_data, init_j2_data = self.readInitPose(self.path)
        self.init_gripper_matrix = self.pose2matrix(init_gripper_data)
        self.init_object_matrix = self.pose2matrix(init_object_data)####init_object_matrix
        self.init_j1_matrix = self.pose2matrix(init_j1_data)
        self.init_j2_matrix = self.pose2matrix(init_j2_data)

        # 输出并保存齐次矩阵到CSV文件
        self.save_matrices_to_csv()

    def pose2matrix(self, pose):
        translation_vector = pose[0:3]
        rotation_vector = pose[3:6]
        rr = spR.from_rotvec(rotation_vector)
        rotation_matrix = rr.as_matrix()
        matrix = np.zeros((4, 4))
        matrix[3, 3] = 1
        matrix[0:3, 0:3] = rotation_matrix
        matrix[0:3, 3] = translation_vector
        return matrix

    def readInitPose(self, init_folder=""):
        file_list = glob.glob(os.path.join(init_folder, "**.txt"))
        print("NUM of .txt file:", len(file_list))

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
    def run(self, dataPath='1.txt'):
        temp = np.loadtxt(dataPath)

        # 检查pose_list的大小
        if len(temp) < 4:
            print(f"Error: Not enough data in pose_list. Expected 4, got {len(temp)}")
            return None, None

        joint_config, objectPose = self.transfer(temp)
        return joint_config, objectPose

    def transfer(self, pose_list=[]):
        # 检查pose_list是否有足够的行数
        if pose_list.shape[0] < 4:
            raise IndexError(f"pose_list has only {pose_list.shape[0]} rows. Expected at least 4.")

        # 保证ID顺序是11, 12, 13, 14
        #pose_list = sorted(pose_list, key=lambda x: x[0])

        gripper_pose = pose_list[2, 1:7]#1x6
        object_pose = pose_list[1, 1:7]#1x6
        gripper_config = np.linalg.norm(pose_list[0, 1:4] - pose_list[3, 1:4])

        griper_matrix = self.pose2matrix(gripper_pose)#4x4
        object_matrix = self.pose2matrix(object_pose)#4x4

        objectInHand = np.matmul(np.linalg.inv(griper_matrix), object_matrix)#4x4,objectInHand：物体相对于夹爪的变换矩阵。
        initObjectInHand = np.matmul(np.linalg.inv(self.init_gripper_matrix), self.init_object_matrix)#4x4,初始状态下（物体未被抓取时）物体相对于夹爪的姿态.

        transfer_matrix = np.matmul(np.linalg.inv(initObjectInHand), objectInHand)#4x4,当前姿态相对于初始姿态的变换

        rr = spR.from_matrix(transfer_matrix[0:3, 0:3])
        rotation_vect = rr.as_rotvec() #旋转向量 (轴角表示)
        euler_angles = rr.as_euler('xyz', degrees=True)#240905 对象rr转换为欧拉角
        #rotation_vect = np.degrees(rotation_vect)#旋转向量 (轴角表示,not as_euler
        return gripper_config * 1000, np.hstack((transfer_matrix[0:3, 3] * 1000, euler_angles))

    def save_matrices_to_csv(self):
        # 将矩阵数据保存到CSV文件中
        matrices = {
            'init_gripper_matrix': self.init_gripper_matrix,
            'init_object_matrix': self.init_object_matrix,
            'init_j1_matrix': self.init_j1_matrix,
            'init_j2_matrix': self.init_j2_matrix
        }

        for name, matrix in matrices.items():
            csv_filename = f'{name}.csv'
            np.savetxt(csv_filename, matrix, delimiter=',')
            print(f"{name} saved to {csv_filename}")

if __name__ == "__main__":
    CAP = SingleVideoCapture(cameraID=8)#1/2
    MarkerDetector = ReadAruCo()

    initial_pose_path = "./IMG_DATA_LS/IMG_DATA_MASTERBALL_0905-183019/Pose_Init"#2/2
    a = RawDataTransfer(initial_pose_path)

    count = 1
    resultpose = []

    while True:
        img = CAP.read()
        result_img, pose_data = MarkerDetector.readPose(img)

        if pose_data:
            detected_ids = [p[0] for p in pose_data]
            required_ids = [11, 12, 13, 14]

            # 检查是否检测到所有四个ID
            if all(req_id in detected_ids for req_id in required_ids):
                # 根据ID排序pose_data
                pose_data_sorted = sorted(pose_data, key=lambda x: required_ids.index(x[0]))

                formatted_data = "\n".join(["{:.1f} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e}".format(
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6]) for p in pose_data_sorted])

                temp_data_path = "temp_pose_data.txt"
                with open(temp_data_path, "w") as temp_file:
                    temp_file.write(formatted_data)
                
                joint_config, GroundTruthPOSE = a.run(temp_data_path)#####runrunrunrunrun

            

                if joint_config is not None and GroundTruthPOSE is not None:

                    GroundTruthPOSE[5] = (GroundTruthPOSE[5] + 360) % 360
                    
                    print("Transformed joint_config:", joint_config)
                    print("Transformed GroundTruthPOSE:", GroundTruthPOSE) 

                    if len(resultpose) == 0:  # 初次赋值
                        resultpose = GroundTruthPOSE
                    else:
                        resultpose = np.concatenate((resultpose, GroundTruthPOSE))  # 注意这里的双括号
                    
                    print("Transformed resultpose:", resultpose)
        else:
            print("Error: Not enough valid pose data detected.")
        
        count += 1
        print("count:", count)

        cv2.imshow('Camera', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 停止摄像头采集
    CAP.stop()
    cv2.destroyAllWindows()

    # 将 resultpose 重整为 nx6 的格式
    resultpose = np.reshape(resultpose, (-1, 6))

    # 保存到 CSV 文件
    csv_filename = 'resultpose.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(resultpose)

    print(f"Data saved to {csv_filename}")

import csv
import matplotlib.pyplot as plt

# 定义数据存储变量
x = []
y = []
z = []
roll = []
pitch = []
yaw = []

# 读取 CSV 文件
file_path = 'resultpose.csv'
with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        # 将字符串转换为浮点数并添加到各自的列表中
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(float(row[2]))
        roll.append(float(row[3]))
        pitch.append(float(row[4]))
        yaw.append(float(row[5]))

# 绘制每个分量
plt.figure(figsize=(12, 8))

# 绘制 x, y, z 位置
plt.subplot(2, 1, 1)
plt.plot(x, label='X Position')
plt.plot(y, label='Y Position')
plt.plot(z, label='Z Position')
plt.title('Position Components Over Time')
plt.xlabel('Count')
plt.ylabel('Position (mm)')
plt.legend()

# 绘制 roll, pitch, yaw 角度
plt.subplot(2, 1, 2)
plt.plot(roll, label='Roll (deg)')
plt.plot(pitch, label='Pitch (deg)')
plt.plot(yaw, label='Yaw (deg)')
plt.title('Orientation Components Over Time')
plt.xlabel('Count')
plt.ylabel('Angle (degrees)')
plt.legend()

plt.tight_layout()

img_folder = '/home/sen/Documents/InHand_pose/seeArucoAccurcyResultpose_plot'
 
if not os.path.isdir(img_folder):
    os.mkdir(img_folder)

# 保存图形为 PNG 文件
# 正确构建图像文件路径
from datetime import datetime
output_file_path = os.path.join(img_folder, 'resultpose_plot_' + datetime.now().strftime("%m%d-%H%M%S") + '.png')
plt.savefig(output_file_path)

plt.show()