# import imp
import os
# from cv2 import waitKey
# import skimage
import glob
import numpy as np
import random
from scipy.spatial.transform import Rotation as spR


current_wd = os.getcwd()
file_path=os.path.abspath(__file__)
print(file_path, os.path.dirname(file_path))
os.chdir(os.path.dirname(file_path))

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
        gripper_list = []
        object_list = []
        joint1 = []
        joint2 = []

        for i in range(len(file_list)):
            temp = np.loadtxt(file_list[i])
            gripper_list.append(temp[0,1:7])
            object_list.append(temp[3,1:7])
            joint1.append(temp[1,1:7])
            joint2.append(temp[2,1:7])
        
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
        gripper_pose = pose_list[0,1:7]
        object_pose = pose_list[3,1:7]

        # 2-finger gripper, 1 dof, config is width between fingers
        gripper_config = np.linalg.norm(pose_list[1,1:4]-pose_list[2,1:4])

        griper_matirx = self.pose2matrix(gripper_pose)
        object_matrix = self.pose2matrix(object_pose)

        # print("griper_matirx:",gripper_pose)
        # print("object_matrix:",object_pose)
        # # 
        objectInHand = np.matmul(np.linalg.inv(griper_matirx), object_matrix)
        initObjectInHand = np.matmul(np.linalg.inv(self.init_gripper_matrix), self.init_object_matrix)

        # print("objectInHand:",objectInHand)
        # print("initObjectInHand inv:",np.linalg.inv(initObjectInHand))

        # transfer_matrix  = np.matmul(objectInHand, np.linalg.inv(initObjectInHand))
        transfer_matrix  = np.matmul(np.linalg.inv(initObjectInHand), objectInHand)
        # print("transfer_matrix:",transfer_matrix)

        rr = spR.from_matrix(transfer_matrix[0:3, 0:3])
        rotation_vect = rr.as_rotvec()
        return gripper_config*1000, np.hstack((transfer_matrix[0:3,3]*1000, rotation_vect))

    def run(self, dataPath='1.txt'):
        temp = np.loadtxt(dataPath)
        # TODO:
        # if len(temp.shape)==1:
        #     print('yy:', temp, dataPath)
        joint_config, objectPose = self.transfer(temp)
        return joint_config, objectPose


if __name__ == "__main__":
    file_folder = "./IMG_DATA_tri/POSE_TXT/*.txt"
    file_list = glob.glob(file_folder)
    # print(file_list[0])
    object_list = []
    for i in range(len(file_list)):
        temp = np.loadtxt(file_list[i])
        object_list.append(temp[3,1:7])
    object_list = np.array(object_list)
    # print(object_list)
    print(np.max(object_list, axis=0)-np.min(object_list, axis=0))
    print(np.max(object_list, axis=0))
    print(np.min(object_list, axis=0))


    init_folder = "./IMG_DATA_tri/Pose_Init"
    pr = RawDataTransfer(init_folder)

    transfer_pose = []
    for i in range(len(file_list)):
        joint_config, objectPose = pr.run(file_list[i])
        transfer_pose.append(objectPose)
        # if objectPose[0]>235:
        #     print(i, objectPose, file_list[i])

    transfer_pose = np.array(transfer_pose)
    print("========================================================")
    print(np.max(transfer_pose, axis=0)-np.min(transfer_pose, axis=0))
    print(np.max(transfer_pose, axis=0))
    print(np.min(transfer_pose, axis=0))

    # 4689, 
    # joint_config, objectPose = pr.run(file_list[23])
    # temp = np.loadtxt(file_list[4689])

    # print("transferd pose:",objectPose)
    # print(temp[3,1:7])









