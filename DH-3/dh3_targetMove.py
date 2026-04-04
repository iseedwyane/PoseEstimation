from ControlGripper_DH3 import SetCmd
import time





def move_gripper_to_position(target_position):
    gripper = SetCmd()

    time.sleep(1)

    # 获取当前位置（这里假设为50，您可能需要根据实际情况调整或动态获取）
    start_position = 0
    gripper.Position(start_position)


    # 确保目标位置在合理范围内
    target_position = max(0, min(target_position, 95))

    # 计算需要循环的次数，步长为1
    for count in range(abs(target_position - start_position)):
        # 设置夹爪的新位置
        if target_position > start_position:
            new_position = start_position + count
        else:
            new_position = start_position - count

        gripper.Position(new_position)
        time.sleep(0.05)  # 短暂暂停确保移动平滑

    # 最后设置夹爪到目标位置
    gripper.Position(target_position)
    move_gripper_to_position(75)

    
    
