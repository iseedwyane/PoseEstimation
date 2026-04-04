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
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.flag = False
        self.cap.release()

class ReadAruCo:
    # Read Aruco pose from image
    def __init__(self) -> None:
        matrix = np.load('camera_params_outside1920x1080LiS.npz')
        self.camera_matrix = matrix['mtx']
        self.camera_dist = matrix['dist']
        
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

        if len(temp) < 4:
            print(f"Error: Not enough data in pose_list. Expected 4, got {len(temp)}")
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
        return gripper_config * 1000, np.hstack((transfer_matrix[0:3, 3] * 1000, rotation_vect))

def initialize_kalman():
    stateSize = 6  # x, y, z, roll, pitch, yaw
    measSize = 6   # measurements for x, y, z, roll, pitch, yaw
    kf = cv2.KalmanFilter(stateSize, measSize)
    
    # 状态转移矩阵 A（假设匀速模型，即位置的变化只取决于速度）
    kf.transitionMatrix = np.eye(stateSize, dtype=np.float32)
    
    # 测量矩阵 H
    kf.measurementMatrix = np.eye(measSize, stateSize, dtype=np.float32)

    # 过程噪声协方差矩阵 Q
    kf.processNoiseCov = np.eye(stateSize, dtype=np.float32) * 1e-2

    # 测量噪声协方差矩阵 R
    kf.measurementNoiseCov = np.eye(measSize, dtype=np.float32) * 1e-1

    # 后验错误估计协方差矩阵 P
    kf.errorCovPost = np.eye(stateSize, dtype=np.float32)

    return kf

def apply_kalman_filter(kf, pose_data):
    # 初始化卡尔曼滤波器的状态和测量
    state = np.zeros((6, 1), np.float32)  # [x, y, z, roll, pitch, yaw]
    meas = np.zeros((6, 1), np.float32)   # [z_x, z_y, z_z, z_roll, z_pitch, z_yaw]
    found = False

    if pose_data:
        meas[0, 0] = pose_data[0][1]  # x
        meas[1, 0] = pose_data[0][2]  # y
        meas[2, 0] = pose_data[0][3]  # z
        meas[3, 0] = pose_data[0][4]  # roll
        meas[4, 0] = pose_data[0][5]  # pitch
        meas[5, 0] = pose_data[0][6]  # yaw

        if not found:
            kf.errorCovPre = np.eye(6, dtype=np.float32)
            state = meas.copy()
            kf.statePost = state
            found = True
        else:
            kf.correct(meas)

    # 使用卡尔曼滤波器预测新的状态
    prediction = kf.predict()
    return prediction



if __name__ == "__main__":
    CAP = SingleVideoCapture(cameraID=0)
    MarkerDetector = ReadAruCo()

    initial_pose_path = "./IMG_DATA_LS/IMG_DATA_MASTERBALL_0830/Pose_Init"
    a = RawDataTransfer(initial_pose_path)

    count = 1
    resultpose = []
    filtered_resultpose = []

    kf = initialize_kalman()  # 初始化卡尔曼滤波器

    while True:
        img = CAP.read()
        result_img, pose_data = MarkerDetector.readPose(img)

        if pose_data:
            detected_ids = [p[0] for p in pose_data]
            required_ids = [11, 12, 13, 14]

            if all(req_id in detected_ids for req_id in required_ids):
                pose_data_sorted = sorted(pose_data, key=lambda x: required_ids.index(x[0]))

                formatted_data = "\n".join(["{:.1f} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e}".format(
                    p[0], p[1], p[2], p[3], p[4], p[5], p[6]) for p in pose_data_sorted])

                temp_data_path = "temp_pose_data.txt"
                with open(temp_data_path, "w") as temp_file:
                    temp_file.write(formatted_data)
                
                joint_config, GroundTruthPOSE = a.run(temp_data_path)
                if joint_config is not None and GroundTruthPOSE is not None:
                    print("Transformed joint_config:", joint_config)
                    print("Transformed GroundTruth pose:", GroundTruthPOSE) 

                    if len(resultpose) == 0:
                        resultpose = GroundTruthPOSE
                    else:
                        resultpose = np.concatenate((resultpose, GroundTruthPOSE))
                    
                    print("Transformed resultpose:", resultpose)

                    # 应用卡尔曼滤波器
                    filtered_pose = apply_kalman_filter(kf, pose_data)
                    print("Transformed filtered_pose:", filtered_pose)
                    # 将 filtered_pose 转换为一维数组
                    filtered_pose_flat = filtered_pose.flatten()
                    filtered_resultpose = np.concatenate((filtered_resultpose, filtered_pose_flat))
                    
        else:
            print("Error: Not enough valid pose data detected.")
        
        count += 1
        print("count:", count)

        cv2.imshow('Camera', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CAP.stop()
    cv2.destroyAllWindows()

    resultpose = np.reshape(resultpose, (-1, 6))
    filtered_resultpose = np.reshape(filtered_resultpose, (-1, 6))

    csv_filename = 'resultpose.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(resultpose)

    filtered_csv_filename = 'filtered_resultpose.csv'
    with open(filtered_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(filtered_resultpose)

    print(f"Data saved to {csv_filename}")
    print(f"Filtered data saved to {filtered_csv_filename}")
