import rtde_control
import rtde_receive
from PyQt5.QtCore import QThread
from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从模块中直接导入需要的类
import time
import serial

class RobotThread(QThread):
    def __init__(self, ip_address="192.168.1.102"):
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

home =   [-0.08899991614701225, -0.5894864793845794, 0.6237713944898031, -0.21856300810944504, 3.132459038531462, 0.011655843025216786]

approach = [-0.08899991614701225, -0.5894864793845794,0.4478657888310303, -0.21855277043854626, 3.1324432510823312, 0.01163866327081556]

# 创建机器人线程对象
robot = RobotThread()
gripper = SetCmd() 
time.sleep(1)
gripper.Position(95)#1/1
#gripper.angle(60)
gripper.Force(15)

# 读取并打印当前的TCP位置
current_pose = robot.get_current_pose()
print("Current TCP Pose:", current_pose)
# 控制机器人移动到目标位置
robot.move(home, speed=0.05, acceleration=0.5, async_move=False)

gripper.Position(80)#1/1
#gripper.angle(60)
gripper.Force(15)
# 读取并打印当前的TCP位置
current_pose = robot.get_current_pose()
print("Current TCP Pose:", current_pose)

# 控制机器人移动到另一个目标位置
robot.move(approach, speed=0.05, acceleration=0.3, async_move=True)
gripper.Position(95)#1/1
#gripper.angle(60)
gripper.Force(15)



pick =  [-0.08899991614701225, -0.5894864793845794,0.4357121285267734, -0.21857695537779417, 3.1324275522604865, 0.011655059065769073]

robot.move(pick, speed=0.05, acceleration=0.3, async_move=False)
step=1

gripper.Position(60)#1/1
#gripper.angle(60)
gripper.Force(15)

goup = [-0.08899991614701225, -0.5894864793845794, 0.6237597523355269, -0.21856070325557883, 3.1324484321255337, 0.011679185957335343]
robot.move(goup, speed=0.05, acceleration=0.3, async_move=False)
print("UR10 & DH3 done")
