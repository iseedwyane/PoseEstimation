import pygame
import serial
import time

# 打开串口通信
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # 修改为正确的串口名

def map_axis_value(value, reverse=False):
    # 将 [-1, 1] 映射到 [0, 255]
    if reverse:
        return int((1 - value) * 127.5)
    else:
        return int((value + 1) * 127.5)

def send_data_to_arduino(mapped_left_x, mapped_left_y, mapped_right_y):
    # 发送三个数值，格式为: "mapped_left_x,mapped_left_y,mapped_right_y\n"
    send_data = f"{mapped_left_x},{mapped_left_y},{mapped_right_y}\n"
    ser.write(send_data.encode())  # 将数据发送到 Arduino
    print(f"Sent to Arduino: {send_data}")
    
    # 读取 Arduino 的反馈
    time.sleep(0.1)  # 给 Arduino 一些处理时间
    while ser.in_waiting > 0:
        feedback = ser.readline().decode().strip()
        print(f"Arduino says: {feedback}")

# 假设这些值是从游戏手柄或其他传感器读取的映射值
mapped_left_x = map_axis_value(0.5)  # 示例映射值
mapped_left_y = map_axis_value(-0.2)  # 示例映射值
mapped_right_y = map_axis_value(0.7, reverse=True)  # 示例映射值

# 将数据发送到 Arduino
#send_data_to_arduino(mapped_left_x, mapped_left_y, mapped_right_y)

# 关闭串口
#ser.close()



def get_data_from_joyStick():
    pygame.init()
    pygame.joystick.init()

    joystick = pygame.joystick.Joystick(0)  # 使用第一个游戏手柄
    joystick.init()

    dead_zone = 0.1

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # 获取左摇杆的 X 和 Y 轴值
        axis_left_x = joystick.get_axis(0)  # 左摇杆 X 轴
        axis_left_y = joystick.get_axis(1)  # 左摇杆 Y 轴

        # 获取右摇杆的 Y 轴值
        axis_right_y = joystick.get_axis(4)  # 右摇杆 Y 轴

        # 处理死区
        axis_left_x = 0 if abs(axis_left_x) < dead_zone else axis_left_x
        axis_left_y = 0 if abs(axis_left_y) < dead_zone else axis_left_y
        axis_right_y = 0 if abs(axis_right_y) < dead_zone else axis_right_y

        # 将左摇杆从 [-1, 1] 映射到 [0, 255]
        mapped_left_x = map_axis_value(axis_left_x)
        mapped_left_y = map_axis_value(axis_left_y, reverse=True)

        # 将右摇杆 Y 轴从 [-1, 1] 反转映射到 [255, 0]
        mapped_right_y = map_axis_value(axis_right_y, reverse=True)
        print(mapped_left_x,mapped_left_y,mapped_right_y)
        # 构造发送的数据字符串
        #send_data = f"{mapped_left_x},{mapped_left_y},{mapped_right_y}\n"
        send_data_to_arduino(mapped_left_x, mapped_left_y, mapped_right_y)
        time.sleep(0.25) #key
        # 控制输出频率，每秒更新20次
        pygame.time.Clock().tick(20)

    pygame.quit()
    ser.close()

# 调用摇杆读取函数
get_data_from_joyStick()
