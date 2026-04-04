import cv2
import numpy as np
import queue
import threading
import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import serial
import time

class SingleVideoCapture:
    """ Single camera video capture management class. """
    def __init__(self, cameraID=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(cameraID)
        if not self.cap.isOpened():
            raise Exception(f"Camera {cameraID} could not be opened.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q = queue.Queue(maxsize=10)
        self.flag = True

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while self.flag:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame.")
                break
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            frame = self.q.get(timeout=1)
            return frame
        except queue.Empty:
            print("Queue is empty, no frame to retrieve.")
            return None

    def stop(self):
        self.flag = False
        self.cap.release()

class ReadAruCo:
    """ Aruco code pose reader using a single camera with PD control. """
    def __init__(self, g_ref, Kp, Kd, ser):
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.mark_size = 0.011
        self.frame_count = 0
        self.g_ref = g_ref  # Target height
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.ser = ser
        self.prev_error = 0.0
        self.prev_time = datetime.now()

    def readPose(self, img, save_folder):
        if img is None:
            print("Image is None.")
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        if ids is None:
            print("No Aruco IDs detected.")
            return img

        img_with_aruco = img.copy()
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)

        for i in range(len(ids)):
            if ids[i] == 12:  # Process only marker with ID 12
                img_with_aruco = cv2.aruco.drawAxis(img_with_aruco, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
                img_with_aruco = cv2.aruco.drawDetectedMarkers(img_with_aruco, corners, ids)
                print(f"ID 12: Rotation Vector (rvec) = {rvec[i]}, Translation Vector (tvec) = {tvec[i]}")
                print(f"ID 12: Translation Vector (tvec[i][0][1]) = {tvec[i][0][1]}")
                # Calculate the current height and error
                tvec[i][0][1] = -tvec[i][0][1]
                g = tvec[i][0][1] - 0.0  # Adjust height
                error = (self.g_ref - g)*1000
                current_time = datetime.now()
                dt = (current_time - self.prev_time).total_seconds()
                d_error = (error - self.prev_error) / dt if dt > 0 else 0

                # Debug prints to check error values and PWM adjustments
                print(f"Current Height: {g:.2f}, Error: {error:.2f}, D_Error: {d_error:.2f}")

                pwm = self.Kp * error + self.Kd * d_error
                pwm = max(0, min(255, int(pwm)))  # Ensure PWM is within [0, 255]

                # Send PWM value to Arduino
                self.send_to_arduino(pwm)

                # Save data with timestamp
                timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                file_name_txt = "{:08d}.txt".format(self.frame_count)
                file_path = os.path.join(save_folder, file_name_txt)
                with open(file_path, 'w') as f:
                    f.write(f"{timestamp},{g}\n")
                self.frame_count += 1
                self.prev_error = error
                self.prev_time = current_time

        return img_with_aruco

    def send_to_arduino(self, pwm_value):
        self.ser.write(f"{pwm_value}\n".encode())
        print(f"Sent PWM value: {pwm_value}")

def save_height_data_to_csv(data_folder, csv_filename):
    timestamps = []
    heights = []
    filenames = sorted([f for f in os.listdir(data_folder) if f.endswith('.txt')])

    for filename in filenames:
        file_path = os.path.join(data_folder, filename)
        try:
            with open(file_path, 'r') as f:
                line = f.readline().strip()
                timestamp, height = line.split(',')
                timestamps.append(timestamp)
                heights.append(float(height))
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")

    if not heights:
        print("No height data found to save.")
        return

    base_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
    relative_times = [(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') - base_time).total_seconds() for ts in timestamps]

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'Height (m)'])
        for rel_time, height in zip(relative_times, heights):
            writer.writerow([rel_time, height])
    print(f"Height data saved to {csv_filename}.")

def plot_height_changes(data_folder):
    timestamps = []
    heights = []
    filenames = sorted([f for f in os.listdir(data_folder) if f.endswith('.txt')])

    for filename in filenames:
        file_path = os.path.join(data_folder, filename)
        try:
            with open(file_path, 'r') as f:
                line = f.readline().strip()
                timestamp, height = line.split(',')
                timestamps.append(timestamp)
                heights.append(float(height))
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")

    if not heights:
        print("No height data found to plot.")
        return

    base_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
    relative_times = [(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') - base_time).total_seconds() for ts in timestamps]

    plt.figure()
    plt.plot(relative_times, heights, marker='o', label='Height (m)')
    plt.title('Height Changes Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(data_folder, 'Height_Changes_Over_Time.png'))
    plt.show()


if __name__ == "__main__":
    img_folder = "./DH-3/ArcucoPOSE"
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    hightpose_folder = f"HightPose_{timestamp}"

    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    hightpose_path = os.path.join(img_folder, hightpose_folder)
    if not os.path.isdir(hightpose_path):
        os.mkdir(hightpose_path)

    video_capture = SingleVideoCapture(cameraID=4)

    # 设置目标高度、PD 增益和串口通信
    g_ref = 0.005  # 设定的目标高度
    Kp = 100.0   # 比例增益
    Kd = 50.0    # 微分增益
    ser = serial.Serial('/dev/ttyUSB0', 115200)  # 根据需要调整端口

    aruco_reader = ReadAruCo(g_ref, Kp, Kd, ser)

    while True:
        img = video_capture.read()
        if img is None:
            print("No frame retrieved.")
            continue
        result_img = aruco_reader.readPose(img, hightpose_path)
        cv2.imshow('Camera', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.stop()
    cv2.destroyAllWindows()

    # 保存高度数据到 CSV
    csv_filename = os.path.join(hightpose_path, 'height_data.csv')
    save_height_data_to_csv(hightpose_path, csv_filename)

    # 绘制高度变化图
    plot_height_changes(hightpose_path)

