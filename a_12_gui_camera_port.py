from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys, cv2, numpy as np
from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从模块中直接导入需要的类
import time
import serial
####DH-3
gripper = SetCmd() 
time.sleep(1)
step=1

gripper.Position(70)
gripper.angle(60)
gripper.Force(40)


class VideoThread(QThread):
    # 定义信号，用于发送捕获到的帧
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.port = self.find_camera_port()

    def find_camera_port(self):
        # 尝试打开不同的端口以找到连接的摄像头
        for port in range(50):
            cap = cv2.VideoCapture(port)
            if cap.isOpened():
                cap.release()
                return port
        return None

    def run(self):
        if self.port is None:
            print("No camera found!")
            return

        # 打开摄像头
        self.cap = cv2.VideoCapture(self.port)
        while True:
            ret, cv_img = self.cap.read()
            cv_img = np.rot90(cv_img, 1) 
            if ret:
                # 发射信号
                self.change_pixmap_signal.emit(cv_img)

    def stop(self):
        # 停止摄像头捕获
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()

class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()
        
        
        # 创建视频线程对象
        self.thread = VideoThread()
        # 连接信号与槽
        self.thread.change_pixmap_signal.connect(self.update_image)

        self.initUI()
        # 开始线程
        self.thread.start()
        
        # 创建一个定时器，用于自动更新状态
        self.timer = QTimer(self)
        self.timer.setInterval(1000)  # 设置时间间隔为1000毫秒（1秒）
        self.timer.timeout.connect(self.update_status)
        self.timer.start()


    def initUI(self):
        # 设置窗口的尺寸和标题
        self.setGeometry(200, 200, 1200, 800)
        self.setWindowTitle('Simple PyQt5 App with Camera')

        # 使用布局管理器来组织控件
        layout = QVBoxLayout(self)

        # 创建一个用于显示视频的 QLabel
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        layout.addWidget(self.image_label)

        # 创建显示当前相机端口号的标签
        self.camera_port_label = QLabel(f'Camera Port: {self.thread.port}', self)
        layout.addWidget(self.camera_port_label)

        # 创建用于显示夹爪状态的标签
        self.force_label = QLabel('Force: ', self)
        self.position_label = QLabel('Position: ', self)
        self.angle_label = QLabel('Angle: ', self)
        self.feedback_label = QLabel('Feedback: ', self)
        self.feedback_label.setWordWrap(True)  # 设置文字换行

        # 将状态标签添加到布局中
        layout.addWidget(self.force_label)
        layout.addWidget(self.position_label)
        layout.addWidget(self.angle_label)
        layout.addWidget(self.feedback_label)

        # 创建一个按钮，并设置它的位置和点击事件
        btn = QPushButton('Update Status', self)
        btn.clicked.connect(self.update_status)
        layout.addWidget(btn)

        # 创建输入框和标签
        self.position_input = QLineEdit(self)
        self.position_input.setPlaceholderText("Enter Position (0-95)")
        self.angle_input = QLineEdit(self)
        self.angle_input.setPlaceholderText("Enter Angle (0-100)")
        self.force_input = QLineEdit(self)
        self.force_input.setPlaceholderText("Enter Force (10-90)")

        layout.addWidget(QLabel('Position:'))
        layout.addWidget(self.position_input)
        layout.addWidget(QLabel('Angle:'))
        layout.addWidget(self.angle_input)
        layout.addWidget(QLabel('Force:'))
        layout.addWidget(self.force_input)

        # 创建MOVE按钮
        move_btn = QPushButton('MOVE', self)
        move_btn.clicked.connect(self.move_gripper)
        layout.addWidget(move_btn)
        self.setLayout(layout)


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """用新的 OpenCV 图像更新 QLabel"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """将 OpenCV 图像转换为 QPixmap 格式"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_status(self):
        """读取夹爪状态并更新标签"""
        force_value = gripper.ForceRead()
        self.force_label.setText(f"Force: {force_value}")

        position_value = gripper.PositionRead()
        self.position_label.setText(f"Position: {position_value}")

        angle_value = gripper.angleRead()
        self.angle_label.setText(f"Angle: {angle_value}")

        feedback_value = gripper.FeedbackRead()
        self.feedback_label.setText(f"Feedback: {feedback_value}")
    def move_gripper(self):
        """读取输入框的值并控制夹爪"""
        try:
            position = int(self.position_input.text())
            angle = int(self.angle_input.text())
            force = int(self.force_input.text())

            # 控制夹爪移动
            gripper.Position(position)
            gripper.angle(angle)
            gripper.Force(force)

            # 手动更新状态
            self.update_status()

        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', 'Please enter valid numbers for Position, Angle, and Force.')


def main():
    app = QApplication(sys.argv)
    ex = SimpleApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
