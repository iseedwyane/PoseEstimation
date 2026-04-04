import torch
import cv2
import numpy as np
from PoseNet_4Cam import PNet
from LoadData_4Cam import ReconstDataset
from scipy.spatial.transform import Rotation as spR
from torchvision import transforms
import os
import queue
import threading
import time
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from AutoEncoder import AE
import serial
from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从模块中直接导入需要的类
import glob
import csv

class RealTimePrediction:
    def __init__(self):
        self.pre_model_pth = "./weights/reconstruction_weight_LS_64_3bjects_0830.pth"#1/5
        self.width = 320
        self.height = 320
        self.batch_size = 32

        self.saved_pth = "./weights/pose_weight.pth"
        self.data_folder = "./IMG_DATA"  # 任意路径，因为训练不需要此输入
        self.data_folder_var = "./IMG_DATA"

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def preprocess_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        img = img.astype(np.float32) / 255.0  # 归一化
        img = cv2.resize(img, (self.width, self.height))  # 调整大小
        return img        

class MultiVideoCapture:
    """ 3 cameras"""
    def __init__(self, cameraID_outside=2, cameraID_left=6, cameraID_right=8, cameraID_behind=0, width=640, height=360):
        self.cap_outside = cv2.VideoCapture(cameraID_outside)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap_outside.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.cap_left = cv2.VideoCapture(cameraID_left)
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.cap_right = cv2.VideoCapture(cameraID_right)
        self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
        

    def _reader(self):
        while self.flag:
            ret1, frame_outside = self.cap_outside.read()
            ret2, frame_left = self.cap_left.read()
            ret3, frame_right = self.cap_right.read()
            ret4, frame_behind = self.cap_behind.read()
            if not (ret1 and ret2 and ret3 and ret4):
                break
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

class ReadAruCo:
    def __init__(self):
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
        kernel = np.ones((5,5),np.float32)/25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)

        if ids is None or len(ids) != 4:
            return False, False
        
        indexs = ids.argsort(axis=0).reshape(-1)
        order_ids = ids[indexs]
        orders_corners = np.array(corners)[indexs]
        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(orders_corners, self.mark_size, self.camera_matrix, self.camera_dist)

        for i in range(len(ids)):
            color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
        
        return np.hstack((order_ids, tvec.reshape(-1,3), rvec.reshape(-1,3))), color_image_result


class RawDataTransfer(object):
    def __init__(self, init_path = "") -> None:
        # 读取初始参数姿态
        self.path = init_path

        # 初始姿态
        init_gripper_data, init_object_data, init_j1_data, init_j2_data = self.readInitPose(self.path)
        self.init_gripper_matrix = self.pose2matrix(init_gripper_data)
        self.init_object_matrix = self.pose2matrix(init_object_data)
        self.init_j1_matrix = self.pose2matrix(init_j1_data)
        self.init_j2_matrix = self.pose2matrix(init_j2_data)

    def pose2matrix(self, pose):
        # 6d 位姿转换为4×4矩阵
        transation_vector = pose[0:3]
        rotation_vector = pose[3:6]
        rr = spR.from_rotvec(rotation_vector)
        rotation_matrix = rr.as_matrix()
        matrix = np.zeros((4,4))
        matrix[3,3]=1
        matrix[0:3, 0:3]=rotation_matrix
        matrix[0:3, 3]=transation_vector
        return matrix 

    def readInitPose(self, init_folder=""):
        # 平均姿态作为初始姿态
        file_list = glob.glob(os.path.join(init_folder, "**.txt"))
        print("NUM of .txt file:",len(file_list))
        print("txt file:",file_list)
        #print(file_list)
        gripper_list = []
        object_list = []
        joint1 = []
        joint2 = []

        for i in range(len(file_list)):
            temp = np.loadtxt(file_list[i])
            joint1.append(temp[0,1:7])
            object_list.append(temp[1,1:7])
            gripper_list.append(temp[2,1:7])
            joint2.append(temp[3,1:7])
        
        gripper_list = np.array(gripper_list)
        object_list = np.array(object_list)
        joint1 = np.array(joint1)
        joint2 = np.array(joint2)
        
        # mean rotation vector
        mg = spR.from_rotvec(gripper_list[:, 3:6])
        gripper_r_init = mg.mean().as_rotvec()
        gripper_t_init = np.mean(gripper_list[:, 0:3],axis=0)

        mo = spR.from_rotvec(object_list[:, 3:6])
        object_r_init = mo.mean().as_rotvec()
        object_t_init = np.mean(object_list[:, 0:3],axis=0)

        mj1 = spR.from_rotvec(joint1[:, 3:6])
        mj1_r_init = mj1.mean().as_rotvec()
        mj1_t_init = np.mean(joint1[:, 0:3],axis=0)

        mj2 = spR.from_rotvec(joint2[:, 3:6])
        mj2_r_init = mj2.mean().as_rotvec()
        mj2_t_init = np.mean(joint2[:, 0:3],axis=0)

        return np.hstack((gripper_t_init, gripper_r_init)), np.hstack((object_t_init, object_r_init)), \
            np.hstack((mj1_t_init, mj1_r_init)), np.hstack((mj2_t_init, mj2_r_init))
    
    def transfer(self, pose_list = []):
        # 根据保存的姿态，转换为需要的姿态T
        # TODO:
        # if len(pose_list.shape)==1:
        #     print('xx:',pose_list)
        gripper_pose = pose_list[2,1:7]
        object_pose = pose_list[1,1:7]

        # 2-finger gripper, 1 dof, config is width between fingers
        gripper_config = np.linalg.norm(pose_list[0,1:4]-pose_list[3,1:4])

        griper_matirx = self.pose2matrix(gripper_pose)
        object_matrix = self.pose2matrix(object_pose)

        # 
        objectInHand = np.matmul(np.linalg.inv(griper_matirx), object_matrix)
        initObjectInHand = np.matmul(np.linalg.inv(self.init_gripper_matrix), self.init_object_matrix)

        # transfer_matrix  = np.matmul(objectInHand, np.linalg.inv(initObjectInHand))
        transfer_matrix  = np.matmul(np.linalg.inv(initObjectInHand), objectInHand)

        rr = spR.from_matrix(transfer_matrix[0:3, 0:3])
        rotation_vect = rr.as_rotvec()
        return gripper_config*1000, np.hstack((transfer_matrix[0:3,3]*1000, rotation_vect))
        # return gripper_config*1000, np.hstack((object_pose[0:3]*1000, object_pose[3:6]))
        #return gripper_config*1000,  object_pose[0:3]*1000
        #return gripper_config*1000,  rotation_vect,  

    def run(self, dataPath='1.txt'):
        temp = np.loadtxt(dataPath)
        # TODO:
        # if len(temp.shape)==1:
        #     print('yy:', temp, dataPath)
        joint_config, objectPose = self.transfer(temp)
        return joint_config, objectPose
    
# Serial setup
ser = serial.Serial('/dev/ttyUSB0', 115200)  # Replace with actual port number
def send_pwm_value(pwm_value):
        if -255 <= pwm_value <= 255:
            ser.write(f"{pwm_value}\n".encode())
            print(f"Sent PWM value: {pwm_value}")
        elif pwm_value< -255:
            pwm_value = -255
            ser.write(f"{pwm_value}\n".encode())
            print(f"Sent PWM value: {pwm_value}")
        elif pwm_value> 255:
            pwm_value = 255
            ser.write(f"{pwm_value}\n".encode())
            print(f"Sent PWM value: {pwm_value}")            
        else:
            print("Invalid PWM value, should be between -255 and 255")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CAP = MultiVideoCapture(cameraID_outside=2, cameraID_left=10, cameraID_right=4, cameraID_behind=0)#2/5
    #3/5
    saved_pth = ["./weights/pose_weight_masterball_0830.pth","./weights/pose_weight_bottle_0830.pth","./weights/pose_weight_peach_0830.pth"]#2/4
    #saved_pth = ["./weights/pose_weight_baseball_0807.pth"]    
    weight_path = saved_pth[0]
    print(weight_path)  
    time.sleep(3)
    pre_model_pth = "./weights/reconstruction_weight_LS_64_3bjects_0830.pth"#4/5

    pose_model = PNet(pre_model_pth).to(device)
    pose_model.load_state_dict(torch.load(weight_path))
    pose_model.eval()

    prediction = RealTimePrediction()
    MarkerDetector = ReadAruCo()
    gripper = SetCmd() 
    #PD
    ref = 60
    kp = 8

    i =0
    #current_path = os.getcwd()
    #print("Current working directory:", current_path)
    #file_path = os.path.join(current_path, "IMG_DATA_LS/IMG_DATA_BASEBALL_0628/Pose_Init")
    #current_path = os.getcwd()
    #print("Current working directory:", current_path)   

    file_path = "./IMG_DATA_LS/IMG_DATA_MASTERBALL_0830/Pose_Init"#5/5
    #file_path = "./IMG_DATA_LS/IMG_DATA_BASEBALL_0807-094136/Pose_Init"
    a = RawDataTransfer(file_path)
    init_gripper_data, init_object_data, init_j1_data, init_j2_data  = a.readInitPose(file_path)
    init_gripper_matrix = a.pose2matrix(init_gripper_data)
    init_object_matrix = a.pose2matrix(init_object_data)
    # 创建并打开CSV文件以追加模式写入数据
    with open('output_data'+ datetime.now().strftime("%m%d-%H%M%S")+'.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'Sensoroutput1', 'Sensoroutput2', 'Sensoroutput3', 'output4', 'output5', 'output6', 
                         'GroundTruthPOSE1', 'GroundTruthPOSE2', 'GroundTruthPOSE3', 
                         'GroundTruthPOSE4', 'GroundTruthPOSE5', 'GroundTruthPOSE6'])
            
        try:
            #while True:
            while i<1000:
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]  # 获取当前时间戳

                img_outside, img_left, img_right, img_behind = CAP.read()          
                height, width, channels = img_left.shape

                img1_data = prediction.preprocess_image(img_left)
                img2_data = prediction.preprocess_image(img_right)
                img3_data = prediction.preprocess_image(img_behind)

                result_pose, result_img = MarkerDetector.readPose(img_outside)
                #print("result_pose:", result_pose)
                if result_pose is not False:
                    # 识别ID为11和14的二维码，并计算它们之间的距离
                    pose_11 = None
                    pose_14 = None
                    #transfer
                    
                    for pose in result_pose:
                        if pose[0] == 11:
                            pose_11 = pose[1:4]  # 提取位置信息
                        elif pose[0] == 14:
                            pose_14 = pose[1:4]  # 提取位置信息
                        elif pose[0] ==13:
                            gripper_pose = pose[1:7]
                        elif pose[0] == 12:
                            object_pose = pose[1:7]
                        else:
                            print("result_pose met error")

                    # 使用RawDataTransfer类进行坐标系变换
                    joint_config, GroundTruthPOSE = a.run(result_pose)
                    print("Transformed joint_config:", joint_config)
                    print("Transformed GroundTruthPOSE:", GroundTruthPOSE)   
                         
                    griper_matirx = a.pose2matrix(gripper_pose)
                    object_matrix = a.pose2matrix(object_pose)
                    objectInHand = np.matmul(np.linalg.inv(griper_matirx), object_matrix)
                    initObjectInHand = np.matmul(np.linalg.inv(init_gripper_matrix), init_object_matrix)

                    # transfer_matrix  = np.matmul(objectInHand, np.linalg.inv(initObjectInHand))
                    transfer_matrix  = np.matmul(np.linalg.inv(initObjectInHand), objectInHand)  
                    #Object_transferred_position = transfer_matrix[0:3,3]*1000              
                    rr = spR.from_matrix(transfer_matrix[0:3, 0:3])
                    rotation_vect = rr.as_rotvec()
                    #print("Length of rotation_vect:", len(rotation_vect))
                    #print("Data of rotation_vect:", rotation_vect)
                    GroundTruthPOSE = np.hstack((transfer_matrix[0:3,3]*1000, rotation_vect))

                    print("rotation_vect:", rotation_vect)

                    if pose_11 is not None and pose_14 is not None:
                        gripper_config = np.linalg.norm(pose_11 - pose_14)* 1000
                        #print(f"Distance between ID 11 and ID 14: {gripper_config} meters")
                else:
                    gripper_config = 0.1030280649243982 * 1000
                    #print("No ID 12, gripper_config met error")
                #print("gripper_config:", gripper_config)

                imgLeft = torch.Tensor(np.array([img1_data])).to(device)
                imgRight = torch.Tensor(np.array([img2_data])).to(device)
                imgBehind = torch.Tensor(np.array([img3_data])).to(device)
                joint_config = torch.Tensor(np.array([gripper_config])).to(device)

                imgLeft = imgLeft.unsqueeze(0)
                imgRight = imgRight.unsqueeze(0)
                imgBehind = imgBehind.unsqueeze(0)
                joint_config = joint_config.unsqueeze(0)

                outputs = pose_model(imgLeft, imgRight, imgBehind, joint_config)   
                #print("Predicted Pose:", outputs)
                # 获取输出张量的第6个值并转换为标量
                if outputs.numel() > 1:  # 确保outputs有多个元素
                    last_value = outputs[0, 5].item()  # 取第一行的第6个元素
                    last_value = np.rad2deg(last_value) # 将角度转为弧度
                    print("Predict Value:", last_value)


                if outputs.numel() > 1:  # 确保outputs有多个元素
                    output_values = outputs[0].detach().cpu().numpy()  # 转换为numpy数组
                    #print("Predicted Pose:", output_values)
                # 将 GroundTruthPOSE 的姿态数据（旋转向量）从弧度转换为度
                if not all(np.isnan(GroundTruthPOSE)):
                    GroundTruthPOSE[:3] = GroundTruthPOSE[:3]  # 保持位置部分不变
                    GroundTruthPOSE[3:] = np.rad2deg(GroundTruthPOSE[3:])  # 将旋转向量部分转为度
                    
                print("Ground Truth:", GroundTruthPOSE)
                # 写入CSV文件
                writer.writerow([timestamp] + list(output_values) + list(GroundTruthPOSE))


                PWM_PD_value = -(ref - last_value)*kp
                #send_pwm_value(PWM_PD_value)
                
                #Object_size=0.85
                GRIPPER_PD_value = 65
                #print("GRIPPER_Position value:", GRIPPER_PD_value)
                gripper.Position(GRIPPER_PD_value)
                time.sleep(0.25)
                #cv2.imshow(f"Camera ", result_img)
                #print("imshow imshow by LiS")


                #outputs  #sensor
                #GroundTruthPOSE   #Ground Truth
                #time 
                if i == 100:
                    break

                print(i)
                i +=1        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Program interrupted by LiS")
        finally:
            CAP.stop()
            cv2.destroyAllWindows()