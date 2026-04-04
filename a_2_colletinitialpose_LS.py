# MIT License.
# Copyright (c) 2022 by BioicDL. All rights reserved.
# Created by LiuXb on 2022/5/11
# -*- coding:utf-8 -*-

"""
@Modified: 
@Description: collect training img
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
    """ 3 cameras"""
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


if __name__ == "__main__":
    # load img
    CAP = MultiVideoCapture(cameraID_outside=0, cameraID_left=2, cameraID_right=8)
    MarkerDetector = readAruCo()
    img_folder = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_VASE_' + datetime.now().strftime("%m%d-%H%M%S")
    #img_folder = "./IMG_DATA_can"
    #os.mkdir(img_folder)

    # 定义基础目录和将要创建的新目录 
    #base_directory = '/home/sen/Documents/inhandpose/IMG_DATA_LS' 
    #new_directory_name = 'IMG_DATA_CUBE' + datetime.now().strftime("%m%d-%H%M%S") 
    #full_path = os.path.join(base_directory, new_directory_name) # 使用 os.makedirs 创建目录，确保父目录的存在 
    #os.makedirs(full_path, exist_ok=True)

    outside_folder = "IMG_OUTSIDE"
    left_folder = "IMG_LEFT"
    right_folder = "IMG_RIGHT"
    result_folder = "Pose_Init"
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    if not os.path.isdir(os.path.join(img_folder, outside_folder)):
        os.mkdir(os.path.join(img_folder, outside_folder))

    if not os.path.isdir(os.path.join(img_folder, left_folder)):
        os.mkdir(os.path.join(img_folder, left_folder))

    if not os.path.isdir(os.path.join(img_folder, right_folder)):
        os.mkdir(os.path.join(img_folder, right_folder))

    if not os.path.isdir(os.path.join(img_folder, result_folder)):
        os.mkdir(os.path.join(img_folder, result_folder))

    pose_list_L = []
    pose_list_Object = []
    pose_list_Gripper = []
    pose_list_R = []
    
    t1 = time.time()
    for i in range(0, 100):
        img_outside, img_left, img_right, img_hehind = CAP.read()
        #img_outside = np.rot90(img_outside, 1)  

        
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
                elif marker_id == 13:  # 假设ID为13的标记代表Gripper
                    pose_list_Gripper.append(pose)
                elif marker_id == 14:  # 假设ID为14的标记代表Right
                    pose_list_R.append(pose)

            cv2.imshow('overview', result_img)
            cv2.waitKey(1)
            # save data
            file_name_img = "{:08d}.jpg".format(i)
            file_name_txt = "{:08d}.txt".format(i)
            cv2.imwrite(os.path.join(img_folder, outside_folder, file_name_img), img_outside)
            cv2.imwrite(os.path.join(img_folder, left_folder, file_name_img), img_left)
            cv2.imwrite(os.path.join(img_folder, right_folder, file_name_img), img_right)
            np.savetxt(os.path.join(img_folder, result_folder, file_name_txt), result_pose)
            # time.sleep(0.02)
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
    print("Average Pose (gstR):", gstR)