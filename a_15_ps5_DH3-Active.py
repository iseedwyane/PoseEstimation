import pygame
import serial
import time
from ControlGripper_DH3 import SetCmd


#####################################ActiveSPN###########################
# 打开串口通信
#ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # 修改为正确的串口名
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # 修改为正确的串口名

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

#####################################DH-3###########################
# Initialize the gripper and set initial parameters
gripper = SetCmd()
time.sleep(1)
PositionReadvalue = gripper.PositionRead()
print("Position:", PositionReadvalue)

# Initial position, force, and angle
position = PositionReadvalue  # Initial position
position =60
force = 15     # Initial force
angle = 60     # Initial angle

gripper.Position(position)
gripper.angle(angle)
gripper.Force(force)

# Function to smoothly change the position within the range of 0-95
def update_position(direction):
    global position
    if direction == 'tighten':
        position += 1  # Tighten by 1 position
        if position > 95:  # Ensure it doesn't exceed the maximum value (95)
            position = 95
    elif direction == 'loosen':
        position -= 1  # Loosen by 1 position
        if position < 0:  # Ensure it doesn't go below the minimum value (0)
            position = 0
    gripper.Position(position)  # Update the gripper position
    print(f"Position set to: {position}")

# Function to smoothly change the force within the range of 10-90
def update_force(direction):
    global force
    if direction == 'increase':
        force += 5  # Increase force by 5 units
        if force > 90:  # Ensure it doesn't exceed the maximum value (90)
            force = 90
    elif direction == 'decrease':
        force -= 5  # Decrease force by 5 units
        if force < 10:  # Ensure it doesn't go below the minimum value (10)
            force = 10
    gripper.Force(force)  # Update the gripper force
    print(f"Force set to: {force}")

# Function to set angle based on button press
def set_angle(degrees):
    global angle
    angle = degrees
    gripper.angle(angle)
    print(f"Angle set to: {angle} degrees")
#####################################get_data_from_joyStick###########################

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

            # Detect button press for angle control
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 2:  # Triangle button (Button 2) sets angle to 60 degrees
                    set_angle(60)
                elif event.button == 3:  # Square button (Button 3) sets angle to 0 degrees
                    set_angle(0)
                elif event.button == 1:  # Circle button (Button 1) sets angle to 90 degrees
                    set_angle(90)
                elif event.button == 0:  # Cross button (Button 0) exits the program
                    print("Cross (Button 0) pressed, exiting...")
                    done = True  # Exit the loop and quit

        # Read D-pad (hat control) for position and force control
        hat = joystick.get_hat(0)  # Read the D-pad
        if hat == (-1, 0):  # Left on D-pad (loosen position)
            print("Loosen gripper")
            for _ in range(5):  # Loosen 5 steps smoothly
                update_position('loosen')
                #time.sleep(0.1)  # Smooth transition with a short delay
        elif hat == (1, 0):  # Right on D-pad (tighten position)
            print("Tighten gripper")
            for _ in range(5):  # Tighten 5 steps smoothly
                update_position('tighten')
                #time.sleep(0.1)  # Smooth transition with a short delay
        elif hat == (0, 1):  # Up on D-pad (increase force)
            print("Increase gripper force")
            for _ in range(1):  # Increase force by 5
                update_force('increase')
                #time.sleep(0.1)
        elif hat == (0, -1):  # Down on D-pad (decrease force)
            print("Decrease gripper force")
            for _ in range(1):  # Decrease force by 5
                update_force('decrease')
                #time.sleep(0.1)


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
