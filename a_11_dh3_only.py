
from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从模块中直接导入需要的类
import time
import serial
####DH-3
gripper = SetCmd() 
time.sleep(1)
step=1

gripper.Position(50)#1/1
gripper.angle(60)
gripper.Force(15)

ForceReadvalue = gripper.ForceRead()
print("Force:", ForceReadvalue)
#time.sleep(1)
PositionReadvalue = gripper.PositionRead()
print("Position:", PositionReadvalue)

#time.sleep(1)
angleReadvalue = gripper.angleRead()
print("angle:", angleReadvalue)
#time.sleep(1)

FeedbackReadReadvalue = gripper.FeedbackRead()
print("FeedbackRead:", FeedbackReadReadvalue)