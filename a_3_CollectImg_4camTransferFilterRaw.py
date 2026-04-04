import cv2
import os
import queue
import threading
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as spR
from datetime import datetime
import glob
from scipy.stats import zscore

# 滑动平均滤波函数
def smooth_data(data, window_size=5):
    """计算滑动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



# 多摄像头采集类
class MultiVideoCapture:
    """采集多个摄像头的图像"""
    def __init__(self, cameraID_outside=0, cameraID_left=10, cameraID_right=8, cameraID_behind=2, width=640, height=360):
        # 外部摄像头
        self.cap_outside = cv2.VideoCapture(cameraID_outside)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # 左手指摄像头
        self.cap_left = cv2.VideoCapture(cameraID_left)
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 右手指摄像头
        self.cap_right = cv2.VideoCapture(cameraID_right)
        self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 后部摄像头
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

    # 采集图像并保存最新的帧
    def _reader(self):
        while self.flag:
            ret1, frame_outside = self.cap_outside.read()
            ret2, frame_left = self.cap_left.read()
            ret3, frame_right = self.cap_right.read()
            ret4, frame_behind = self.cap_behind.read()

            if not ret1:
                print("Error reading from outside camera.")
            if not ret2:
                print("Error reading from left camera.")
            if not ret3:
                print("Error reading from right camera.")
            if not ret4:
                print("Error reading from behind camera.")

            # 继续程序，跳过错误的帧，而不是直接终止
            if not (ret1 and ret2 and ret3 and ret4):
                continue

            if not self.q_outside.empty():
                try:
                    self.q_outside.get_nowait()
                    self.q_left.get_nowait()
                    self.q_right.get_nowait()
                    self.q_behind.get_nowait()
                except queue.Empty:
                    pass

            self.q_outside.put(frame_outside)
            self.q_left.put(frame_left)
            self.q_right.put(frame_right)
            self.q_behind.put(frame_behind)

    def read(self):
        return self.q_outside.get(), self.q_left.get(), self.q_right.get(), self.q_behind.get()

    def stop(self):
        self.flag = False
        self.cap_outside.release()
        self.cap_left.release()
        self.cap_right.release()
        self.cap_behind.release()

# 保留现有的 ArUco 代码识别功能
class ReadAruCo:
    def __init__(self):
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.014

    def readPose(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        if ids is None:
            return img, []

        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)
        pose_data = []
        for i in range(len(ids)):
            if ids[i] in [11, 12, 13, 14]:
                pose = np.hstack((ids[i], tvec[i].flatten(), rvec[i].flatten()))
                pose_data.append(pose)
            color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
            color_image_result = cv2.aruco.drawDetectedMarkers(color_image_result, corners, ids)
            
        pose_data = sorted(pose_data, key=lambda x: x[0])
        return color_image_result, pose_data

# 保留原有的数据转换功能
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
        if len(temp) < 4:
            return None, None
        joint_config, objectPose = self.transfer(temp)
        return joint_config, objectPose

    def transfer(self, pose_list=[]):
        if pose_list.shape[0] < 4:
            raise IndexError(f"pose_list has only {pose_list.shape[0]} rows. Expected at least 4.")
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
        euler_angles = rr.as_euler('xyz', degrees=True)
        return gripper_config * 1000, np.hstack((transfer_matrix[0:3, 3] * 1000, euler_angles))

if __name__ == "__main__":
    CAP = MultiVideoCapture(cameraID_outside=10, cameraID_left=0, cameraID_right=8, cameraID_behind=2)#4/5
    
    img_folder = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_VASE_250109'#1/5
    outside_folder = "IMG_OUTSIDE"
    left_folder = "IMG_LEFT"
    right_folder = "IMG_RIGHT"
    behind_folder = "IMG_BEHIND"
    result_folder = "POSE_TXT"


    for folder in [outside_folder, left_folder, right_folder, behind_folder, result_folder]:
        os.makedirs(os.path.join(img_folder, folder), exist_ok=True)

    MarkerDetector = ReadAruCo()
    #initial_pose_path = "./IMG_DATA_LS/IMG_DATA_VASE_0109-165346/Pose_Init"#2/5
    #data_transfer = RawDataTransfer(initial_pose_path)

    count = 0
    resultpose = []
    
    # 累计总数据量和有效数据量
    total_data_count = 0
    total_valid_data_count = 0

    total_samples = 500

    for j in range(0,500): #h1, change Rx 

        img_outside, img_left, img_right, img_behind = CAP.read()
        result_img, result_pose = MarkerDetector.readPose(img_outside)
        pose_data = result_pose
        if pose_data:
            detected_ids = [p[0] for p in pose_data]
            required_ids = [11, 12, 13, 14]
            if all(req_id in detected_ids for req_id in required_ids):
                temp_data_path = os.path.join(img_folder, 'temp_pose_data.txt') 
                with open(temp_data_path, "w") as temp_file:
                    formatted_data = "\n".join(["{:.1f} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e}".format(
                        p[0], p[1], p[2], p[3], p[4], p[5], p[6]) for p in pose_data])
                    temp_file.write(formatted_data)
                # 假设pose_data中存储了多个标记的位置信息，ID=12的pose数据
                for pose in pose_data:
                    if pose[0] == 12:  # 检查ID是否为12
                        GroundTruthPOSE = pose[1:]  # 提取ID为12的pose信息，位置和姿态
                        break  # 一旦找到ID=12的pose，跳出循环

                if GroundTruthPOSE is not None:
                    print(f"GroundTruthPOSE for ID=12: {GroundTruthPOSE}")
                    # 可进行其他处理操作，比如姿态的角度转换
                    GroundTruthPOSE[5] = (GroundTruthPOSE[5] + 360) % 360  # 对Yaw角度进行处理
                    print(f"Transformed GroundTruthPOSE: {GroundTruthPOSE}")
                else:
                    print("ID=12 pose data not found.")

                    #joint_config, GroundTruthPOSE = data_transfer.run(temp_data_path)
                    if GroundTruthPOSE is not None:
                        GroundTruthPOSE[5] = (GroundTruthPOSE[5] + 360) % 360
                        print(f"Transformed GroundTruthPOSE: {GroundTruthPOSE}")
                        
                        GroundTruthPOSE = GroundTruthPOSE.reshape(1, -1)

                        if len(resultpose) == 0:
                            resultpose = GroundTruthPOSE
                        else:
                            resultpose = np.concatenate((resultpose, GroundTruthPOSE), axis=0)

                    for i in range(len(pose_data)):
                        if pose_data[i][0] == 12:
                            if GroundTruthPOSE is not None:
                                pose_data[i][1:7] = GroundTruthPOSE        #update transfered data

                file_name_txt = "{:08d}.txt".format(j)
                np.savetxt(os.path.join(img_folder, result_folder, file_name_txt), pose_data)

    





        file_name = "{:08d}.jpg".format(j)
        cv2.imwrite(os.path.join(img_folder, outside_folder, file_name), img_outside)
        cv2.imwrite(os.path.join(img_folder, left_folder, file_name), img_left)
        cv2.imwrite(os.path.join(img_folder, right_folder, file_name), img_right)
        cv2.imwrite(os.path.join(img_folder, behind_folder, file_name), img_behind)

        cv2.imshow('Detected Aruco Markers', result_img)
        cv2.waitKey(1)

        print(f"Saved image set {count + 1}/{total_samples}")
        count += 1

    CAP.stop()
    cv2.destroyAllWindows()

    # 输出总的有效数据百分比
    total_valid_percentage = (total_valid_data_count / total_data_count) * 100
    print(f"Total valid data percentage: {total_valid_percentage:.2f}%")

    resultpose = np.reshape(resultpose, (-1, 6))
    csv_filename = os.path.join(img_folder, 'resultpose.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(resultpose)
    print(f"Data saved to {csv_filename}")

    x, y, z, roll, pitch, yaw = [], [], [], [], [], []

    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))
            roll.append(float(row[3]))
            pitch.append(float(row[4]))
            yaw.append(float(row[5]))

    # Apply smoothing to the data
    x_smoothed = smooth_data(x, window_size=5)
    y_smoothed = smooth_data(y, window_size=5)
    z_smoothed = smooth_data(z, window_size=5)
    roll_smoothed = smooth_data(roll, window_size=5)
    pitch_smoothed = smooth_data(pitch, window_size=5)
    yaw_smoothed = smooth_data(yaw, window_size=5)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x_smoothed, label='X Position')
    plt.plot(y_smoothed, label='Y Position')
    plt.plot(z_smoothed, label='Z Position')
    plt.title('Position Components Over Time')
    plt.xlabel('Count')
    plt.ylabel('Position (mm)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(roll_smoothed, label='Roll (rad)')
    plt.plot(pitch_smoothed, label='Pitch (rad)')
    plt.plot(yaw_smoothed, label='Yaw (rad)')
    plt.title('Orientation Components Over Time')
    plt.xlabel('Count')
    plt.ylabel('Angle (radians)')
    plt.legend()

    plt.tight_layout()

    img_folder_plot = os.path.join(img_folder, 'plots')
    os.makedirs(img_folder_plot, exist_ok=True)

    output_file_path = os.path.join(img_folder_plot, 'resultpose_plot_' + datetime.now().strftime("%m%d-%H%M%S") + '.png')
    plt.savefig(output_file_path)
    plt.show()

    print(f"Plot saved to {output_file_path}")
