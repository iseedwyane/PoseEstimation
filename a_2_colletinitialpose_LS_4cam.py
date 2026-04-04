# -*- coding:utf-8 -*-

"""
@Modified: 
@Description: collect training img with an additional camera (hehind)
"""
import cv2
import os
import queue
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as spR
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from scipy import stats  # 新增导入Z-score的库

class MultiVideoCapture:
    """ 4 cameras"""
    def __init__(self, cameraID_outside=0, cameraID_left=6, cameraID_right=8, cameraID_hehind=10, width=640, height=360):
        #  camera for marker
        self.cap_outside = cv2.VideoCapture(cameraID_outside)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        #  camera for left finger
        self.cap_left = cv2.VideoCapture(cameraID_left)
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #  camera for right finger
        self.cap_right = cv2.VideoCapture(cameraID_right)
        self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #  camera for hehind finger
        self.cap_hehind = cv2.VideoCapture(cameraID_hehind)
        self.cap_hehind.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_hehind.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q_outside = queue.Queue()
        self.q_left = queue.Queue()
        self.q_right = queue.Queue()
        self.q_hehind = queue.Queue()

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
            ret4, frame_hehind = self.cap_hehind.read()  # Added for hehind camera
            if not (ret1 and ret2 and ret3 and ret4):  # Check all camera frames
                break
            if not self.q_outside.empty():
                try:
                    self.q_outside.get_nowait()   # discard previous (unprocessed) frame
                    self.q_left.get_nowait()
                    self.q_right.get_nowait()
                    self.q_hehind.get_nowait()  # Added for hehind camera
                except queue.Empty:
                    pass
            self.q_outside.put(frame_outside)
            self.q_left.put(frame_left)
            self.q_right.put(frame_right)
            self.q_hehind.put(frame_hehind)  # Added for hehind camera

    def read(self):
        return self.q_outside.get(), self.q_left.get(), self.q_right.get(), self.q_hehind.get()  # Include hehind camera

    def stop(self):
        self.flag = False
        self.cap_outside.release()
        self.cap_left.release()
        self.cap_right.release()
        self.cap_hehind.release()  # Release hehind camera

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

    def readPose(self, img):
        # return tuple(A,B): A is data, B is image
        #  A: [[marker_id, x,y,z,rx,ry,rz]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)

        if ids is None or len(ids) != 4:
            return False, False

        indexs = ids.argsort(axis=0)
        indexs = indexs.reshape(-1)

        order_ids = ids[indexs]
        corners = np.array(corners)
        orders_corners = corners[indexs]
        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(orders_corners, self.mark_size, self.camera_matrix, self.camera_dist)

        # 输出排序后的id顺序
        print("Detected Aruco IDs in order:", order_ids.flatten())

        for i in range(len(ids)):
            color_image_result = cv2.aruco.drawAxis(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
        return np.hstack((order_ids, tvec.reshape(-1, 3), rvec.reshape(-1, 3))), color_image_result

def calculate_average_pose(pose_list):
    """计算平均姿态"""
    pose_array = np.array(pose_list)
    mean_translation = np.mean(pose_array[:, 1:4], axis=0)
    mean_rotation_vector = np.mean(pose_array[:, 4:7], axis=0)
    return np.hstack((pose_array[0, 0], mean_translation, mean_rotation_vector))

# 新增Z-score异常检测函数
def z_score_filtering(pose_data, threshold=3):
    valid_data = []
    invalid_count = 0
    pose_array = np.array(pose_data)

    # 对每个位置和旋转分量进行Z-score检测
    for i in range(1, 7):  # x, y, z, roll, pitch, yaw
        z_scores = np.abs(stats.zscore(pose_array[:, i]))
        for j in range(len(z_scores)):
            if z_scores[j] <= threshold:
                valid_data.append(pose_array[j])  # 保留有效数据
            else:
                invalid_count += 1  # 统计被剔除的异常数据
    return np.array(valid_data), invalid_count

if __name__ == "__main__":
    # 加载图片
    CAP = MultiVideoCapture(cameraID_outside=4, cameraID_left=0, cameraID_right=2, cameraID_hehind=10)#1/3
    MarkerDetector = readAruCo()
    img_folder = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_VASE_' + datetime.now().strftime("%m%d-%H%M%S")#2/3

    outside_folder = "IMG_OUTSIDE"
    left_folder = "IMG_LEFT"
    right_folder = "IMG_RIGHT"
    behind_folder = "IMG_BEHIND"  # Added folder for hehind camera
    result_folder = "Pose_Init"

    # 新增POSE_init_filtering文件夹
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    pose_filtering_folder = 'POSE_init_filtering'
    if not os.path.isdir(os.path.join(img_folder, pose_filtering_folder)):
        os.mkdir(os.path.join(img_folder, pose_filtering_folder))



    if not os.path.isdir(os.path.join(img_folder, outside_folder)):
        os.mkdir(os.path.join(img_folder, outside_folder))

    if not os.path.isdir(os.path.join(img_folder, left_folder)):
        os.mkdir(os.path.join(img_folder, left_folder))

    if not os.path.isdir(os.path.join(img_folder, right_folder)):
        os.mkdir(os.path.join(img_folder, right_folder))

    if not os.path.isdir(os.path.join(img_folder, behind_folder)):  # Create folder for hehind images
        os.mkdir(os.path.join(img_folder, behind_folder))

    if not os.path.isdir(os.path.join(img_folder, result_folder)):
        os.mkdir(os.path.join(img_folder, result_folder))

    pose_list_L = []
    pose_list_Object = []
    pose_list_Gripper = []
    pose_list_R = []
    
    t1 = time.time()
    for i in range(0, 100):
        img_outside, img_left, img_right, img_hehind = CAP.read()

        result_pose, result_img = MarkerDetector.readPose(img_outside)
        print(i)

        if type(result_pose) == bool:
            continue
        else:
            # 根据id将姿态分配到对应的列表中
            for pose in result_pose:
                marker_id = pose[0]
                if marker_id == 11:  # 假设ID为11的标记代表Left
                    pose_list_L.append(pose)
                elif marker_id == 12:  # 假设ID为12的标记代表Object
                    pose_list_Object.append(pose)
                elif marker_id == 8:  # 假设ID为13的标记代表Gripper #4/4
                    pose_list_Gripper.append(pose)
                elif marker_id == 14:  # 假设ID为14的标记代表Right
                    pose_list_R.append(pose)

            cv2.imshow('overview', result_img)
            cv2.waitKey(1)
            # 保存数据
            file_name_img = "{:08d}.jpg".format(i)
            file_name_txt = "{:08d}.txt".format(i)
            cv2.imwrite(os.path.join(img_folder, outside_folder, file_name_img), img_outside)
            cv2.imwrite(os.path.join(img_folder, left_folder, file_name_img), img_left)
            cv2.imwrite(os.path.join(img_folder, right_folder, file_name_img), img_right)
            cv2.imwrite(os.path.join(img_folder, behind_folder, file_name_img), img_hehind)  # Save hehind image
            np.savetxt(os.path.join(img_folder, result_folder, file_name_txt), result_pose)

    t2 = time.time()
    print("fps:", 1000 / (t2 - t1))

    # 计算每个姿态的平均值
    gstL = calculate_average_pose(pose_list_L)
    gObject = calculate_average_pose(pose_list_Object)
    gGripper = calculate_average_pose(pose_list_Gripper)
    gstR = calculate_average_pose(pose_list_R)

    print("Average Pose (gstL):", gstL)
    print("Average Pose (gObject):", gObject)
    print("Average Pose (gGripper):", gGripper)
    print("Average Pose (gstR) :", gstR)

    # 将 resultpose 重整为 nx6 的格式
    resultpose = np.reshape(pose_list_Object, (-1, 7))
    # 保存到 CSV 文件
    csv_filename = 'Pose_Init.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(resultpose)

    print(f"Data saved to {csv_filename}")    

    # 读取数据并进行Z-score检测
    pose_list_Object_filtered, invalid_Object = z_score_filtering(pose_list_Object)
    total_data_count = len(pose_list_Object)
    valid_data_percentage = (total_data_count - invalid_Object) / total_data_count * 100
    print(f"有效数据百分比: {valid_data_percentage:.2f}%")

    # 保存过滤后的数据
    def save_filtered_pose_data(pose_list, file_prefix):
        for i, pose in enumerate(pose_list):
            file_name_txt = f"{i:08d}.txt"
            np.savetxt(os.path.join(img_folder, pose_filtering_folder, file_name_txt), pose)

    #pose_list_Object_filtered = pose_list_Object_filtered.reshape(1, -1)#
    print("====",pose_list_Object_filtered.shape)
    save_filtered_pose_data(pose_list_Object_filtered, 'Object')

    # 绘制每个分量
    x = []
    y = []
    z = []
    roll = []
    pitch = []
    yaw = []

    # 读取 CSV 文件
    #file_path = os.path.join(img_folder, 'Pose_Init.csv')
    file_path = 'Pose_Init.csv'
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            x.append(float(row[1]))
            y.append(float(row[2]))
            z.append(float(row[3]))
            roll.append(float(row[4]))
            pitch.append(float(row[5]))
            yaw.append(float(row[6]))

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

    img_folder = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_VASE_PlotPose_Init'#3/3
    
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    # 保存图形为 PNG 文件
    output_file_path = os.path.join(img_folder, 'resultpose_plot_' + datetime.now().strftime("%m%d-%H%M%S") + '.png')
    plt.savefig(output_file_path)

    plt.show()
