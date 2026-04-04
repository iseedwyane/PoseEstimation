# MIT License.
# Copyright (c) 2024 by BioicDL. All rights reserved.
# Created by LiS on 2024/9/3
# -*- coding:utf-8 -*-

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
        self.cap = cv2.VideoCapture(cameraID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q = queue.Queue()
        self.flag = True

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

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
        if not self.q.empty():
            return self.q.get()
        else:
            print("Warning: No frame available in the queue.")
            return None

    def stop(self):
        self.flag = False
        self.cap.release()

class ReadAruCo:
    def __init__(self) -> None:
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
        if ids is None:
            return img, None

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
        self.init_object_matrix = self.pose2matrix(init_object_data)
        self.init_j1_matrix = self.pose2matrix(init_j1_data)
        self.init_j2_matrix = self.pose2matrix(init_j2_data)

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

        gripper_pose = pose_list[2, 1:7]
        object_pose = pose_list[1, 1:7]
        gripper_config = np.linalg.norm(pose_list[0, 1:4] - pose_list[3, 1:4])

        griper_matrix = self.pose2matrix(gripper_pose)
        object_matrix = self.pose2matrix(object_pose)

        objectInHand = np.matmul(np.linalg.inv(griper_matrix), object_matrix)
        initObjectInHand = np.matmul(np.linalg.inv(self.init_gripper_matrix), self.init_object_matrix)

        transfer_matrix = np.matmul(np.linalg.inv(initObjectInHand), objectInHand)

        rr = spR.from_matrix(transfer_matrix[0:3, 0:3])
        rotation_vect = rr.as_rotvec()
        return gripper_config * 1000, np.hstack((transfer_matrix[0:3, 3] * 1000, rotation_vect))


if __name__ == "__main__":
    CAP = SingleVideoCapture(cameraID=10)
    MarkerDetector = ReadAruCo()

    initial_pose_path = "./IMG_DATA_LS/IMG_DATA_MASTERBALL_0830/Pose_Init"
    a = RawDataTransfer(initial_pose_path)

    while True:
        img = CAP.read()
        if img is None:
            print("Error: Unable to read frame from camera.")
            continue

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

                joint_config, GroundTruthPOSE = a.run(temp_data_path)
                if joint_config is not None and GroundTruthPOSE is not None:
                    print("Transformed joint_config:", joint_config)
                    print("Transformed GroundTruthPOSE:", GroundTruthPOSE)
            else:
                print("Error: Not all required IDs detected. Skipping frame.")
        else:
            print("Error: Not enough valid pose data detected.")

        cv2.imshow('Camera', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CAP.stop()
    cv2.destroyAllWindows()