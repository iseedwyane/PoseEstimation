import serial
import time

# Serial setup
ser = serial.Serial('/dev/ttyUSB0', 115200)  # Replace with actual port number

def send_pwm_value(pwm_value):
    if -255 <= pwm_value <= 255:
        ser.write(f"{pwm_value}\n".encode())
        print(f"Sent PWM value: {pwm_value}")
    else:
        print("Invalid PWM value, should be between -255 and 255")

try:
    while True:
        # From 0 to -255
        for value in range(0, -256, -5):
            send_pwm_value(value)
            time.sleep(0.1)  # 每2秒发送一次

        # From -255 to 255
        for value in range(-255, 256, 5):
            send_pwm_value(value)
            time.sleep(0.1)  # 每2秒发送一次

        # From 255 to 0
        for value in range(255, -1, -5):
            send_pwm_value(value)
            time.sleep(0.1)  # 每2秒发送一次

except KeyboardInterrupt:
    ser.close()
    print("Program stopped")
