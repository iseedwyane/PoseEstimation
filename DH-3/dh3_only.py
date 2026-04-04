
from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从模块中直接导入需要的类
import time
import serial
####DH-3
gripper = SetCmd() 
time.sleep(1)
step=1
gripper.Position(40)