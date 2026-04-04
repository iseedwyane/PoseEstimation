# -*- coding:utf-8 -*-

"""
@Modified: 
@Description: collect training img with an additional camera (behind) without data modification
"""
import cv2
import os
import queue
import threading
import time
import numpy as np
from datetime import datetime
import csv
import matplotlib.pyplot as plt

class MultiVideoCapture:
    """ 4 cameras"""
    def __init__(self, cameraID_outside=0, cameraID_left=6, cameraID_right=8, cameraID_behind=10, width=640, height=360):
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
        #  camera for behind finger
        self.cap_behind = cv2.VideoCapture(cameraID_behind)
        self.cap_behind.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_behind.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q_outside = queue.Queue()
        self.q_left = queue.Queue()
        self.q_right = queue.Queue()
        self.q_behind = queue.Queue()

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
            ret4, frame_behind = self.cap_behind.read()  # Added for behind camera
            if not (ret1 and ret2 and ret3 and ret4):  # Check all camera frames
                break
            if not self.q_outside.empty():
                try:
                    self.q_outside.get_nowait()   # discard previous (unprocessed) frame
                    self.q_left.get_nowait()
                    self.q_right.get_nowait()
                    self.q_behind.get_nowait()  # Added for behind camera
                except queue.Empty:
                    pass
            self.q_outside.put(frame_outside)
            self.q_left.put(frame_left)
            self.q_right.put(frame_right)
            self.q_behind.put(frame_behind)  # Added for behind camera

    def read(self):
        return self.q_outside.get(), self.q_left.get(), self.q_right.get(), self.q_behind.get()  # Include behind camera

    def stop(self):
        self.flag = False
        self.cap_outside.release()
        self.cap_left.release()
        self.cap_right.release()
        self.cap_behind.release()  # Release behind camera

"""read aruco code pose"""
class readAruCo():
    # read Aruco pose from image
    def __init__(self) -> None:
        # camera 0
        self.camera_matrix = np.array([[1.35248113e+03, 0.00000000e+00, 1.14469132e+03],
                                       [0.00000000e+00, 1.35890869e+03, 4.98796253e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([-0.3869634 ,  0.21638318  ,0.00060695 ,-0.01747466 ,-0.11219537])

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

# 滑动平均函数
def moving_average(data, window_size=5):
    """计算滑动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

if __name__ == "__main__":
    # 加载图片
    CAP = MultiVideoCapture(cameraID_outside=8, cameraID_left=6, cameraID_right=4, cameraID_behind=10)  # 1/3
    MarkerDetector = readAruCo()
    img_folder = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_SP_' + datetime.now().strftime("%m%d-%H%M%S")  # 2/3

    outside_folder = "IMG_OUTSIDE"
    left_folder = "IMG_LEFT"
    right_folder = "IMG_RIGHT"
    behind_folder = "IMG_BEHIND"  # Added folder for behind camera
    result_folder = "Pose_TXT"

    # 创建文件夹
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

    pose_list_L = []
    pose_list_Object = []
    pose_list_Gripper = []
    pose_list_R = []

    # 设置视频保存参数
    video_filename = os.path.join(img_folder, 'output_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码
    fps = 30  # 设置帧率
    frame_width = 1920  # 视频宽度
    frame_height = 1080  # 视频高度
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    t1 = time.time()
    for i in range(0, 99):
        img_outside, img_left, img_right, img_behind = CAP.read()

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
            # 保存数据
            file_name_img = "{:08d}.jpg".format(i)
            file_name_txt = "{:08d}.txt".format(i)
            cv2.imwrite(os.path.join(img_folder, outside_folder, file_name_img), img_outside)
            cv2.imwrite(os.path.join(img_folder, left_folder, file_name_img), img_left)
            cv2.imwrite(os.path.join(img_folder, right_folder, file_name_img), img_right)
            cv2.imwrite(os.path.join(img_folder, behind_folder, file_name_img), img_behind)  # Save behind image
            np.savetxt(os.path.join(img_folder, result_folder, file_name_txt), result_pose)

            # 写入视频帧
            out.write(result_img)  # 将当前帧写入视频文件

    # 释放视频文件
    out.release()

    t2 = time.time()
    print("fps:", 1000 / (t2 - t1))

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

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

    # 绘制每个分量
    x = []
    y = []
    z = []
    roll = []
    pitch = []
    yaw = []

    # 读取 CSV 文件
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

    # 对数据进行滑动平均
    window_size = 5  # 可调整窗口大小
    x_smooth = moving_average(x, window_size)
    y_smooth = moving_average(y, window_size)
    z_smooth = moving_average(z, window_size)
    roll_smooth = moving_average(roll, 10)
    pitch_smooth = moving_average(pitch, window_size)
    yaw_smooth = moving_average(yaw, 10)

    plt.figure(figsize=(12, 8))

    # 绘制 x, y, z 位置
    plt.subplot(2, 1, 1)
    plt.plot(x_smooth, label='X Position')
    plt.plot(y_smooth, label='Y Position')
    plt.plot(z_smooth, label='Z Position')
    plt.title('Position Components Over Time (Smoothed)')
    plt.xlabel('Count')
    plt.ylabel('Position (mm)')
    plt.legend()

    # 绘制 roll, pitch, yaw 角度
    plt.subplot(2, 1, 2)
    plt.plot(roll_smooth, label='Roll (deg)')
    plt.plot(pitch_smooth, label='Pitch (deg)')
    plt.plot(yaw_smooth, label='Yaw (deg)')
    plt.title('Orientation Components Over Time (Smoothed)')
    plt.xlabel('Count')
    plt.ylabel('Angle (degrees)')
    plt.legend()

    plt.tight_layout()

    img_folder = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_SP_PlotPose_Init'  # 3/3

    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    # 保存图形为 PNG 文件
    output_file_path = os.path.join(img_folder, 'resultpose_plot_' + datetime.now().strftime("%m%d-%H%M%S") + '.png')
    plt.savefig(output_file_path)

    plt.show()
