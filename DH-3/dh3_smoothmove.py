from ControlGripper_DH3 import SetCmd
import time

gripper = SetCmd()

time.sleep(1)

# 初始位置和结束位置
start_position = 95
gripper.Position(start_position)
time.sleep(2)  # 短暂暂停确保移动平滑
end_position = 50
step = 5  # 设置步长为5

# 计算需要循环的次数
for count in range(0, end_position - start_position, step):
    # 设置夹爪的新位置
    new_position = start_position + count
    gripper.Position(new_position)
    #time.sleep(0.1)  # 短暂暂停确保移动平滑

# 确保最后夹爪到达结束位置
gripper.Position(end_position)