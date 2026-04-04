import serial
import time

# 串口设置
ser = serial.Serial('/dev/ttyUSB0', 115200)  # 替换为实际的端口号

def send_pwm_value(pwm_value):
    if -255 <= pwm_value <= 255:
        ser.write(f"{pwm_value}\n".encode())
        print(f"Sent PWM value: {pwm_value}")
    else:
        print("Invalid PWM value, should be between 0 and 255")

try:
    while True:
        # 示例：逐渐增加 PWM 值
        for value in range(-255, 255, 25):
            send_pwm_value(value)
            time.sleep(2)  # 每2秒发送一次

except KeyboardInterrupt:
    ser.close()
    print("Program stopped")
