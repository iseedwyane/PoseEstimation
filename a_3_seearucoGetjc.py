# MIT License.
# Copyright (c) 2022 by BioicDL. All rights reserved.
# Created by LiuXb on 2022/5/11
# -*- coding:utf-8 -*-

"""
@Modified: 
@Description: Display Aruco code detection from a single camera
"""
import cv2
import numpy as np


class SingleVideoCapture:
    """ Single camera """
    def __init__(self, cameraID=4, width=1920, height=1080):
        # Camera for marker
        self.cap = cv2.VideoCapture(cameraID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


"""Read Aruco code pose"""
class ReadAruCo:
    # Read Aruco pose from image
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
            return False, img

        indexs = ids.argsort(axis=0).reshape(-1)
        order_ids = ids[indexs]
        orders_corners = np.array(corners)[indexs]
        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(orders_corners, self.mark_size, self.camera_matrix, self.camera_dist)

        for i in range(len(ids)):
            color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
        
        return np.hstack((order_ids, tvec.reshape(-1, 3), rvec.reshape(-1, 3))), color_image_result


if __name__ == "__main__":
    CAP = SingleVideoCapture(cameraID=12)
    MarkerDetector = ReadAruCo()

    while True:
        img = CAP.read()
        if img is None:
            print("Failed to grab frame")
            break

        result_pose, result_img = MarkerDetector.readPose(img)
        if result_pose is not False:
            # 识别ID为11和14的二维码，并计算它们之间的距离
            pose_11 = None
            pose_14 = None

            for pose in result_pose:
                if pose[0] == 11:
                    pose_11 = pose[1:4]  # 提取位置信息
                elif pose[0] == 14:
                    pose_14 = pose[1:4]  # 提取位置信息

            if pose_11 is not None and pose_14 is not None:
                distance = np.linalg.norm(pose_11 - pose_14)
                print(f"Distance between ID 11 and ID 14: {distance} meters")

        cv2.imshow('Camera', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CAP.release()
    cv2.destroyAllWindows()