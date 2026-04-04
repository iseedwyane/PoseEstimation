# -*- coding:utf-8 -*-
"""
@Modified: 
@Description: collect training img with behind camera
"""

import cv2
import os
import queue
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as spR
from datetime import datetime

class MultiVideoCapture:
    """
    4 cameras: outside, left, right, behind
    多摄像头采集类，负责同时从4个摄像头中采集图像。
    摄像头分别用于外部环境、左手指、右手指和背后视角的图像采集。
    """
    def __init__(self, cameraID_outside=0, cameraID_left=6, cameraID_right=8, cameraID_behind=10, width=640, height=360):
        # 初始化外部摄像头
        self.cap_outside = cv2.VideoCapture(cameraID_outside)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # 初始化左手指摄像头
        self.cap_left = cv2.VideoCapture(cameraID_left)
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 初始化右手指摄像头
        self.cap_right = cv2.VideoCapture(cameraID_right)
        self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 初始化背后摄像头
        self.cap_behind = cv2.VideoCapture(cameraID_behind)
        self.cap_behind.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_behind.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 创建队列用于存储摄像头的图像帧
        self.q_outside = queue.Queue()
        self.q_left = queue.Queue()
        self.q_right = queue.Queue()
        self.q_behind = queue.Queue()

        self.flag = True

        # 启动后台线程采集图像
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # 从摄像头读取图像帧，并只保留最新的图像
    def _reader(self):
        while self.flag:
            # 从四个摄像头读取图像帧
            ret1, frame_outside = self.cap_outside.read()
            ret2, frame_left = self.cap_left.read()
            ret3, frame_right = self.cap_right.read()
            ret4, frame_behind = self.cap_behind.read()

            # 如果读取失败，则退出循环
            if not (ret1 and ret2 and ret3 and ret4):
                break

            # 丢弃旧的帧，只保留最新的帧
            if not self.q_outside.empty():
                try:
                    self.q_outside.get_nowait()   # discard previous (unprocessed) frame
                    self.q_left.get_nowait()
                    self.q_right.get_nowait()
                    self.q_behind.get_nowait()
                except queue.Empty:
                    pass

            # 将最新的图像帧放入队列
            self.q_outside.put(frame_outside)
            self.q_left.put(frame_left)
            self.q_right.put(frame_right)
            self.q_behind.put(frame_behind)

    # 获取最新的图像帧
    def read(self):
        return self.q_outside.get(), self.q_left.get(), self.q_right.get(), self.q_behind.get()

    # 停止摄像头采集，并释放资源
    def stop(self):
        self.flag = False
        self.cap_outside.release()
        self.cap_left.release()
        self.cap_right.release()
        self.cap_behind.release()


"""read aruco code pose"""
class readAruCo():
    """
    从图像中检测Aruco码并计算其位姿信息（位置和旋转）。
    """
    def __init__(self) -> None:
        # 初始化相机内参矩阵和畸变参数
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])
        
        # 初始化Aruco码字典和检测参数
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.016  # Aruco码的物理尺寸

    # 从输入图像中检测Aruco码的位姿
    def readPose(self, img):
        """
        输入：图像img
        输出：
        - 位姿数据：marker_id, x, y, z, 旋转向量(rx, ry, rz)
        - 包含检测到的Aruco码坐标轴的图像
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)

        # 如果没有检测到4个Aruco码，则返回False
        if ids is None or len(ids) != 4:
            return False, False

        # 按照ID顺序对Aruco码进行排序
        indexs = ids.argsort(axis=0)
        indexs = indexs.reshape(-1)
        order_ids = ids[indexs]
        corners = np.array(corners)
        orders_corners = corners[indexs]
        color_image_result = img.copy()

        # 估计Aruco码的位姿（位置和旋转）
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(orders_corners, self.mark_size, self.camera_matrix, self.camera_dist)

        # 在图像中绘制Aruco码的坐标轴
        for i in range(len(ids)):
            color_image_result = cv2.aruco.drawAxis(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
        
        # 返回检测到的位姿数据和处理后的图像
        return np.hstack((order_ids, tvec.reshape(-1, 3), rvec.reshape(-1, 3))), color_image_result


if __name__ == "__main__":
    # 初始化摄像头
    CAP = MultiVideoCapture(cameraID_outside=8, cameraID_left=6, cameraID_right=4, cameraID_behind=10)#3/3
    MarkerDetector = readAruCo()

    # 设置图像保存路径
    img_folder = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_SP_0802' #1/3

    outside_folder = "IMG_OUTSIDE"
    left_folder = "IMG_LEFT"
    right_folder = "IMG_RIGHT"
    behind_folder = "IMG_BEHIND"  # Added folder for behind camera
    result_folder = "POSE_TXT"

    # 检查文件夹是否存在，若不存在则创建文件夹
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    if not os.path.isdir(os.path.join(img_folder, outside_folder)):
        os.mkdir(os.path.join(img_folder, outside_folder))

    if not os.path.isdir(os.path.join(img_folder, left_folder)):
        os.mkdir(os.path.join(img_folder, left_folder))

    if not os.path.isdir(os.path.join(img_folder, right_folder)):
        os.mkdir(os.path.join(img_folder, right_folder))

    if not os.path.isdir(os.path.join(img_folder, behind_folder)):  # Create folder for behind images
        os.mkdir(os.path.join(img_folder, behind_folder))

    if not os.path.isdir(os.path.join(img_folder, result_folder)):
        os.mkdir(os.path.join(img_folder, result_folder))

    t1 = time.time()

    # 以下是多种范围采集的循环注释，这些注释用于选择不同采集范围
    # for i in range(0,500):
    # for i in range(500,1000):
    # for i in range(1000,1500):
    # for i in range(1500,2000):
    # for i in range(2000,2500):
    # for i in range(2500,3000):
    # for i in range(3000,3500):
    # for i in range(3500,4000):
    # for i in range(4000,4500):
    # for i in range(4500,5000):
    #for i in range(0,2000): #2/2
    # for i in range(1000,2000):
    # for i in range(2000,3000):
    # for i in range(3000,4000):
    for i in range(2000,5000):  # 当前设置为从4000到5000范围进行采集
        img_outside, img_left, img_right, img_behind = CAP.read()  # 从4个摄像头读取图像
        result_pose, result_img = MarkerDetector.readPose(img_outside)  # 读取Aruco码的位姿信息
        print(i)

        if type(result_pose) == bool:
            continue
        else:
            # 显示处理后的图像
            cv2.imshow('overview', result_img)
            cv2.waitKey(1)

            # 保存图像和位姿数据
            file_name_img = "{:08d}.jpg".format(i)
            file_name_txt = "{:08d}.txt".format(i)
            cv2.imwrite(os.path.join(img_folder, outside_folder, file_name_img), img_outside)
            cv2.imwrite(os.path.join(img_folder, left_folder, file_name_img), img_left)
            cv2.imwrite(os.path.join(img_folder, right_folder, file_name_img), img_right)
            cv2.imwrite(os.path.join(img_folder, behind_folder, file_name_img), img_behind)  # Save behind image
            np.savetxt(os.path.join(img_folder, result_folder, file_name_txt), result_pose)

    t2 = time.time()
    print("fps:", 1000 / (t2 - t1))
