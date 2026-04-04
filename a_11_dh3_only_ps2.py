import serial
import time
from ControlGripper_DH3 import SetCmd  # 从模块中直接导入需要的类

# 设置串口参数
arduino_port = '/dev/ttyUSB0'  
baud_rate = 115200     # 请确保与 Arduino 代码中的波特率一致
timeout = 1            # 设置读取超时时间

# 初始化串口连接
ser = serial.Serial(arduino_port, baud_rate, timeout=timeout)
time.sleep(2)  # 给串口一些时间来进行初始化

print("Starting to read data from Arduino...")

# 初始化DH-3机械手
gripper = SetCmd()
time.sleep(1)
step = 1

# 设置初始位置
current_position = 60  # 初始化位置
gripper.Position(current_position)
gripper.angle(60)
gripper.Force(10)

try:
    while True:
        # 读取从串口发送来的数据
        if ser.in_waiting > 0:
            ps2_rx_value = int(ser.readline().decode('utf-8').strip())
            # 打印接收到的数据
            print(f"Received PSS_RX value: {ps2_rx_value}")

            
            # 将ps2_rx_value限制在-5到5之间
            if ps2_rx_value < -5:
                ps2_rx_value = -5
            elif ps2_rx_value > 5:
                ps2_rx_value = 5

            # 计算新的位置，确保在0到90之间
            new_position = current_position + ps2_rx_value
            new_position = max(0, min(90, new_position))  # 限制位置在0到90之间
            
            # 更新机械手位置
            gripper.Position(new_position)
            time.sleep(1)
            current_position = new_position  # 更新当前位置变量

            # 打印当前状态
            print(f"Updated Position: {new_position}")

            # 获取机械手的当前状态
            ForceReadvalue = gripper.ForceRead()
            #print("Force:", ForceReadvalue)
            
            PositionReadvalue = gripper.PositionRead()
            #print("Position:", PositionReadvalue)
            
            angleReadvalue = gripper.angleRead()
            #print("angle:", angleReadvalue)
            
            FeedbackReadReadvalue = gripper.FeedbackRead()
            #print("FeedbackRead:", FeedbackReadReadvalue)

        time.sleep(0.1)  # 小延迟，防止过度读取

except KeyboardInterrupt:
    print("Exiting program.")

finally:
    ser.close()
    print("Serial connection closed.")
