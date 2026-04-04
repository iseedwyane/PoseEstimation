
import socket, struct, time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import cv2, os, yaml
from ipywidgets import IntProgress
from IPython.display import display
from sklearn import metrics
import pandas as pd
from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从模块中直接导入需要的类

import rtde_control
import rtde_receive
from PyQt5.QtCore import *

class RobotThread(QThread):
    def __init__(self):
        super().__init__()
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
    
    def move(self,target_pose):
        self.rtde_c.moveL(target_pose, 0.01, 0.5)
        
robot = RobotThread()
#time.sleep(2)
gripper = SetCmd() 

robot.rtde_r.getActualTCPPose()

#time.sleep(2)

home =[0.11918188494643467,
 -0.6965688884573296,
 0.40,
1.615929778849063,
 -2.6922346628474596,
 -0.07489480106170308]

#step 1, go home 
target_pose = home
robot.rtde_c.moveL(target_pose, 0.01, 0.5)
time.sleep(3)
#time.sleep(1)



#step 2, go target_pose ############################ exp object choice 
ball_pick_pos = [0.11918107763838645,
 -0.6965871965677204,
 0.35,
1.615929778849063,
 -2.6922346628474596,
 -0.07489480106170308]

target_pose = ball_pick_pos
robot.rtde_c.moveL(target_pose, 0.01, 0.5)
time.sleep(2)
gripper.Position(90)
time.sleep(3)


##ball_gripper = 60
start_position = 90
gripper.Position(start_position)
time.sleep(2)  # 短暂暂停确保移动平滑
end_position = 45
step = 5  # 设置步长为5

# 计算需要循环的次数
for count in range(0, end_position - start_position, step):
    # 设置夹爪的新位置
    new_position = start_position + count
    gripper.Position(new_position)
    #time.sleep(0.1)  # 短暂暂停确保移动平滑

# 确保最后夹爪到达结束位置
gripper.Position(end_position)

time.sleep(2) 

#step 3, go manipulation pos
#gripper.Position(600) 

manipulation = [0.11916907661046584,
 -0.6965999072520557,
 0.38,
1.615929778849063,
 -2.6922346628474596,
 -0.07489480106170308]

target_pose = manipulation
robot.rtde_c.moveL(target_pose, 0.01, 0.5)

#step 4, DH release




#step 5, go home
target_pose = home
#robot.rtde_c.moveL(target_pose, 0.01, 0.5)




