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

        #self.camera_matrix = np.array( [[3.27355836e+03, 0.00000000e+00 ,9.61295920e+02],
        #                                [ 0.00000000e+00, 3.36785229e+03, 5.40250513e+02],
        #                               [  0.     ,      0.       ,    1.        ]])
        #self.camera_dist = np.array([-2.25757211e+00,  1.38323860e+01,  1.13281584e-01, -6.95743965e-02, -8.93461943e+01])


        self.camera_matrix = np.array([[1.21295999e+03, 0.00000000e+00, 9.38139579e+02],
                                       [0.00000000e+00, 1.20761844e+03, 5.55924237e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([-0.34674076,  0.20816553, -0.00061761,  0.00234987, -0.10212255])

        #self.camera_matrix = np.array([[1.19628292e+03, 0.00000000e+00 ,9.26887769e+02],
        #[0.00000000e+00, 1.19432358e+03, 5.20482688e+02],
        #[0., 0., 1.]
        #])
        #self.camera_dist = np.array([-0.34377434,  0.17989047 , 0.00097867 , 0.00115137, -0.07656848])
        
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
            AD = np.cross(AC_norm, AB_norm)

            print("向量AB：", AB_norm)
            print("向量AC：", AC_norm)
            print("向量AD：", AD)
            print("向量AD's norm：", np.linalg.norm(AD))

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
