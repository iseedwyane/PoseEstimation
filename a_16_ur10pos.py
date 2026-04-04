import rtde_control
import rtde_receive
from PyQt5.QtCore import QThread
from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从模块中直接导入需要的类
import time
import serial
from datetime import datetime


class RobotThread(QThread):
    def __init__(self, ip_address="192.168.1.12"):
        super().__init__()
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)
        self.rtde_c = rtde_control.RTDEControlInterface(ip_address)

    def move(self, target_pose, speed=0.05, acceleration=0.5, async_move=False):
        """
        Moves the robot to the target pose using moveL command.
        :param target_pose: list of 6 floats representing the target pose [x, y, z, rx, ry, rz]
        :param speed: movement speed in m/s
        :param acceleration: acceleration in m/s^2
        :param async_move: boolean indicating whether the movement should be asynchronous
        """
        self.rtde_c.moveL(target_pose, speed, acceleration, async_move)

    def get_current_pose(self):
        """
        Retrieves the current position of the robot's TCP.
        :return: list of 6 floats representing the current pose [x, y, z, rx, ry, rz]
        """
        return self.rtde_r.getActualTCPPose()

# 定义目标位置
home =  [-0.15335917846799674, -0.6762462162688213, 0.6455970165526864, -0.11113800587856436, -3.0885142326880928, -0.06819264793829703]

approach = [-0.1533587432227741, -0.6762394993927388, 0.5515089514163866, 0.01899081510444407, 1.6121068345462601, 0.09425134058717571]

# 创建机器人线程对象
robot = RobotThread()
gripper = SetCmd() 
time.sleep(1)

# 读取并打印当前的TCP位置
current_pose = robot.get_current_pose()
print("Current TCP Pose:", current_pose)
# 获取当前时间并格式化为时间戳
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


robot.move(home, speed=0.05, acceleration=0.3, async_move=False)
print(f"Starting时间戳: {timestamp}")

#robot.move(approach, speed=0.05, acceleration=0.3, async_move=False)
# 获取当前时间并格式化为时间戳
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Ending时间戳: {timestamp}")