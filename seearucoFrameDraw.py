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
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])

        self.camera_matrix = np.array([[1.21295999e+03, 0.00000000e+00, 9.38139579e+02],
                                       [0.00000000e+00, 1.20761844e+03, 5.55924237e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([-0.34674076,  0.20816553, -0.00061761,  0.00234987, -0.10212255])
        
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
            return img

        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)

        positions = {}
        for i in range(len(ids)):
            color_image_result = cv2.aruco.drawAxis(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
            color_image_result = cv2.aruco.drawDetectedMarkers(color_image_result, corners, ids)
            if ids[i][0] in [0, 1, 2]:
                positions[ids[i][0]] = tvec[i][0]

        if 0 in positions and 1 in positions and 2 in positions:
            A = positions[0]
            B = positions[1]
            C = positions[2]
            AB = B - A
            AC = C - A
            AB_norm = AB / np.linalg.norm(AB)
            AC_norm = AC / np.linalg.norm(AC)
            AD = np.cross(AC_norm,AB_norm)

            print("向量AB：", AB_norm)
            print("向量AC：", AC_norm)
            print("向量AD：", AD)

            # 绘制向量AB，AC，AD
            start_point = tuple(A[:2].astype(int))
            end_point_AB = tuple((A + AB_norm * 100)[:2].astype(int))
            end_point_AC = tuple((A + AC_norm * 100)[:2].astype(int))
            end_point_AD = tuple((A + AD * 100)[:2].astype(int))

            cv2.arrowedLine(color_image_result, start_point, end_point_AB, (0, 255, 0), 2, tipLength=0.3) # 绿色 AB
            cv2.arrowedLine(color_image_result, start_point, end_point_AC, (255, 0, 0), 2, tipLength=0.3) # 蓝色 AC
            cv2.arrowedLine(color_image_result, start_point, end_point_AD, (0, 0, 255), 2, tipLength=0.3) # 红色 AD

            # 在图像上显示坐标
            cv2.putText(color_image_result, 'A', start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(color_image_result, 'B', end_point_AB, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(color_image_result, 'C', end_point_AC, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(color_image_result, 'D', end_point_AD, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 将坐标转换到世界坐标系
            print(f"R, _ = cv2.Rodrigues(rvec[0]){rvec[0]}")
            R, _ = cv2.Rodrigues(rvec[0])
            T = tvec[0].reshape(3, 1)
            RT = np.hstack((R, T))
            RT = np.vstack((RT, [0, 0, 0, 1]))  # 创建4x4齐次变换矩阵

            # 将A、B、C点转换到世界坐标系
            A_hom = np.append(A, 1)  # 将A点转换为齐次坐标
            A_world = np.dot(RT, A_hom)

            print(f"世界坐标系中的A点：{A_world[:3]}")

        return color_image_result

if __name__ == "__main__":
    CAP = SingleVideoCapture(cameraID=0)
    MarkerDetector = ReadAruCo()

    while True:
        img = CAP.read()
        result_img = MarkerDetector.readPose(img)
        result_img = np.rot90(result_img, 1)    # 对图像矩阵顺时针旋转90度
        cv2.imshow('Camera', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CAP.stop()
    cv2.destroyAllWindows()

    