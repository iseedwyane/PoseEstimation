from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从模块中直接导入需要的类
import time
import serial
####DH-3
gripper = SetCmd() 

time.sleep(1)
gripper.angle(60)
time.sleep(1)
step=1

gripper.Position(45)

I = 200

### 
# 创建串口对象，指定端口和波特率
ser = serial.Serial('/dev/ttyUSB0', 115200)  # 替换为您的Arduino端口号

def send_to_arduino(data):
    ser.write((data + '\n').encode())  # 发送数据到Arduino并添加换行符

try:
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()  # 读取一行数据
            print("Received from Arduino:", data)  # 打印接收到的数据

        # 示例发送数据到Arduino
        #send_to_arduino("stop")
        #time.sleep(1)  # 每2秒发送一次数据
        send_to_arduino("up")
        time.sleep(4)  # 每2秒发送一次数据
        send_to_arduino("down")
        time.sleep(4)  # 每2秒发送一次数据
        #send_to_arduino(I)
        #time.sleep(1)

except KeyboardInterrupt:
    ser.close()
    print("Program stopped")

####DH-3
gripper = SetCmd() 
time.sleep(1)
step=1
gripper.Position(80)
