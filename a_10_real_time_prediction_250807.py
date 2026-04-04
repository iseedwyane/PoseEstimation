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
        self.mark_size = 0.022

        self.last_tvec_y = None
        self.last_tvec_z = None      
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

    def readPose(self, img):
        img = cv2.GaussianBlur(img, (3,3), 0) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #binary = cv2.threshold(gray, n, 255, cv2.THRESH_BINARY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(binary, self.arucoDict, parameters=self.arucoParams)
        if ids is None:
            return img

        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)
        print(rvec.shape)
        for i in range(len(ids)):
            # Only process ID 12
            if ids[i][0] == 12:
                color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
                color_image_result = cv2.aruco.drawDetectedMarkers(color_image_result, corners, ids)
                
                # Save data to CSV
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                with open(self.csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        current_time,
                        int(ids[i][0]),
                        float(tvec[i][0][0]),  # tvec_x
                        float(tvec[i][0][1]),  # tvec_y
                        float(tvec[i][0][2]),  # tvec_z
                        float(rvec[i][0][0]),  # rvec_x
                        float(rvec[i][0][1]),  # rvec_y
                        float(rvec[i][0][2])   # rvec_z
                    ])
                
                # Print info to console
                print(f"ID 12 detected - Position: {tvec[i][0]}, Rotation: {rvec[i][0]}")
                self.last_tvec_y = float(tvec[i][0][1])  # 存储y轴位置
                self.last_tvec_z = float(tvec[i][0][2])  # 存储z轴位置
        return binary, color_image_result




import serial
import time

# 初始化串口
ser = serial.Serial('/dev/ttyACM1', 115200, timeout=1)  # 根据你的设备修改端口
time.sleep(2)  # 等待 Arduino 重启

ser.reset_input_buffer()   # 清空接收缓冲区
ser.reset_output_buffer()  # 清空发送缓冲区


def send_pwm_value(pwm_value):
    # 限幅
    pwm_value = max(-255, min(255, pwm_value))
    
    # 发送值
    ser.write(f"{pwm_value}\n".encode())
    print(f"Sent PWM value: {pwm_value}")
    
    # 等待并读取 Arduino 回传
    response = ser.readline().decode().strip()
    print(f"Arduino response: {response}")



if __name__ == "__main__":
    CAP = SingleVideoCapture(cameraID=8)  # Change camera ID as needed
    MarkerDetector = ReadAruCo()
    ref = 0.35*1000
    kp = 2
    try:
        while True:
            img = CAP.read()
            binary, result_img = MarkerDetector.readPose(img)
            cv2.imshow('Camera', result_img)

            if MarkerDetector.last_tvec_y is not None:
                #qprint(f"ID 12当前Z轴位置: {MarkerDetector.last_tvec_z:.4f} 米")
                print(f"ID 12当前y轴位置: {MarkerDetector.last_tvec_y} ")

            PWM_PD_value = int((ref - MarkerDetector.last_tvec_y*1000)*kp)
            print("PWM_PD_value:", PWM_PD_value)        
            #send_pwm_value(PWM_PD_value)   
            #send_pwm_value(PWM_PD_value)
            time.sleep(0.05)  # 可调整发送间隔，避免Arduino处理不过来

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        CAP.stop()
        cv2.destroyAllWindows()
        print(f"Data saved to: {MarkerDetector.csv_file}")





