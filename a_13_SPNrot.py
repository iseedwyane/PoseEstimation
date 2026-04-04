from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer, Qt
import sys
import serial

# Serial setup
ser = serial.Serial('/dev/ttyUSB0', 115200)  # Replace with actual port number

def send_pwm_value(pwm_value):
    pwm_value = max(-255, min(255, pwm_value))  # Clamp PWM value between -255 and 255
    ser.write(f"{pwm_value}\n".encode())
    print(f"Sent PWM value: {pwm_value}")

class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和尺寸
        self.setWindowTitle("Axis Control")
        self.setGeometry(1000, 1000, 1300, 200)
        
        # 初始化按钮按下时间计时器
        self.pwm_value = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.increase_pwm)
        
        # 创建垂直布局
        layout = QVBoxLayout()
        
        # 创建并添加控制Y轴的按钮
        self.y_button = QPushButton("Move Y", self)
        self.y_button.setCheckable(True)
        self.y_button.pressed.connect(self.start_pwm)
        self.y_button.released.connect(lambda: self.stop_pwm('y'))
        layout.addWidget(self.y_button)
        
        # 创建并添加控制Rx轴的按钮
        self.rx_button = QPushButton("Move Rx", self)
        self.rx_button.setCheckable(True)
        self.rx_button.pressed.connect(self.start_pwm)
        self.rx_button.released.connect(lambda: self.stop_pwm('rx'))
        layout.addWidget(self.rx_button)
        
        # 创建并添加控制Ry轴的按钮
        self.ry_button = QPushButton("Move Ry", self)
        self.ry_button.setCheckable(True)
        self.ry_button.pressed.connect(self.start_pwm)
        self.ry_button.released.connect(lambda: self.stop_pwm('ry'))
        layout.addWidget(self.ry_button)
        
        # 设置布局
        self.setLayout(layout)

    def start_pwm(self):
        self.pwm_value = 0
        self.timer.start(50)  # 每50毫秒增加一次PWM值

    def increase_pwm(self):
        if self.pwm_value < 255:
            self.pwm_value += 5  # 每次增加5的PWM值
            print(f"Current PWM value: {self.pwm_value}")

    def stop_pwm(self, axis):
        self.timer.stop()
        pwm_value = self.pwm_value
        if axis == 'y':
            send_pwm_value(pwm_value)  # 发送对应Y轴的PWM值
        elif axis == 'rx':
            send_pwm_value(pwm_value)  # 发送对应Rx轴的PWM值
        elif axis == 'ry':
            send_pwm_value(pwm_value)  # 发送对应Ry轴的PWM值
        self.pwm_value = 0

def main():
    app = QApplication(sys.argv)
    window = SimpleApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()