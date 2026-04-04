# from turtle import position
import numpy as np
from scipy.spatial.transform import Rotation as spR
import glob
import os


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
        #print("RawDataTransfer ====================================:")


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
        #print("RawDataTransfer ====================================:")
        file_list = glob.glob(os.path.join(init_folder, "**.txt"))
        #print("RawDataTransfer ====================================:")
        print("NUM of .txt file:",len(file_list))
        #print("RawDataTransfer ====================================:")

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

        rr = spR.from_matrix(object_matrix[0:3, 0:3])
        rotation_vect = rr.as_rotvec()

        return gripper_config*1000, np.hstack((object_pose[0:3], object_pose[3:6]))
        #object_pose = object_pose
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


if __name__ == "__main__":
    """ load pose"""
    #file_path = "./IMG_DATA_LS/IMG_DATA_tri_baseball_5000/Pose_Init"
    #a = RawDataTransfer(file_path)
    #b = a.readInitPose(file_path)
    #print("readInitPose",b)
    #print(a.run("./IMG_DATA_tri/Pose_Init/00000017.txt"))
    #print("transfer",a.run("./IMG_DATA_LS/IMG_DATA_tri_baseball_5000/POSE_TXT/00000014.txt"))

    file_path = "./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/Pose_Init"
    a = RawDataTransfer(file_path)
    b = a.readInitPose(file_path)
    print("readInitPose",b)


    print(a.run("./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/POSE_TXT/00000017.txt"))
    #print("transfer",a.run("./IMG_DATA_LS/IMG_DATA_BASEBALL_0628-113856/Pose_Init/00000004.txt"))

    #print("transfer",a.run("./IMG_DATA_LS/IMG_DATA_BASEBALL_0628-151138/Pose_Init/00000004.txt"))

    #print("transfer",a.run("./IMG_DATA_LS/IMG_DATA_BASEBALL_0628-151346/Pose_Init/00000004.txt"))

    #joint_config, objectPose = pose_converter.run('./IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/POSE_TXT/00000017.txt')
    
