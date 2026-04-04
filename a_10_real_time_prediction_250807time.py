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
import csv
import os
import time
from datetime import datetime

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
        self.camera_matrix = np.array([[1.35248113e+03, 0.00000000e+00, 1.14469132e+03],
                                     [0.00000000e+00, 1.35890869e+03, 4.98796253e+02],
                                     [0., 0., 1.]])
        self.camera_dist = np.array([-0.3869634, 0.21638318, 0.00060695, -0.01747466, -0.11219537])

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.014
        
        # Create directory for saving data
        self.save_dir = "aruco_data"
        os.makedirs(self.save_dir, exist_ok=True)

        # Create CSV file for ID 12
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(self.save_dir, f"aruco_id12_data_{timestamp}.csv")
        
        # Write CSV header
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "id", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z"])
            
        # Initialize variables to store last detected pose
        self.last_tvec_y = None
        self.last_tvec_z = None

    def readPose(self, img):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        
        # Default values when no marker is detected
        id_found = 0
        tvec_data = [0, 0, 0]
        rvec_data = [0, 0, 0]
        marker_detected = False

        if ids is not None:
            color_image_result = img.copy()
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)

            for i in range(len(ids)):
                # Only process ID 12
                if ids[i][0] == 12:
                    color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
                    color_image_result = cv2.aruco.drawDetectedMarkers(color_image_result, corners, ids)
                    
                    # Update data with detected marker
                    id_found = int(ids[i][0])
                    tvec_data = [float(tvec[i][0][0]), float(tvec[i][0][1]), float(tvec[i][0][2])]
                    rvec_data = [float(rvec[i][0][0]), float(rvec[i][0][1]), float(rvec[i][0][2])]
                    marker_detected = True
                    
                    # Print info to console
                    print(f"ID 12 detected - Position: {tvec_data}, Rotation: {rvec_data}")
                    self.last_tvec_y = float(tvec[i][0][1])  # 存储y轴位置
                    self.last_tvec_z = float(tvec[i][0][2])  # 存储z轴位置
        else:
            color_image_result = img.copy()
            
        # Save data to CSV (whether marker was detected or not)
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_time,
                id_found,
                tvec_data[0],  # tvec_x
                tvec_data[1],  # tvec_y
                tvec_data[2],  # tvec_z
                rvec_data[0],  # rvec_x
                rvec_data[1],  # rvec_y
                rvec_data[2]   # rvec_z
            ])
            
        return color_image_result


if __name__ == "__main__":
    CAP = SingleVideoCapture(cameraID=8)  # Change camera ID as needed
    MarkerDetector = ReadAruCo()
    ref = 0.35*1000
    kp = 20
    try:
        while True:
            img = CAP.read()
            result_img = MarkerDetector.readPose(img)
            cv2.imshow('Camera', result_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        CAP.stop()
        cv2.destroyAllWindows()
        print(f"Data saved to: {MarkerDetector.csv_file}")