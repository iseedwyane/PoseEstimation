from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys, cv2, numpy as np
from ControlGripper_DH3 import SetCmd  # 从模块中直接导入需要的类
import time
import serial

# 创建夹爪控制实例
gripper = SetCmd() 
time.sleep(1)

class VideoThread(QThread):
    # 定义信号，用于发送捕获到的帧
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.port = None

    def detect_cameras(self):
        available_ports = []
        for port in range(50):  # 假设最多有50个摄像头端口
            cap = cv2.VideoCapture(port)
            if cap.isOpened():
                available_ports.append(port)
                cap.release()
        return available_ports

    def select_camera(self, port):
        if self.cap is not None:
            self.cap.release()
        self.port = port
        self.cap = cv2.VideoCapture(self.port)

    def run(self):
        if self.port is None:
            print("No camera found!")
            return

        while True:
            ret, cv_img = self.cap.read()
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
        
        # 初始化UI
        self.initUI()

        # 连接信号与槽
        self.thread.change_pixmap_signal.connect(self.update_image)

        # 检测可用摄像头
        self.available_cameras = self.thread.detect_cameras()
        if self.available_cameras:
            self.camera_selector.addItems([f"Camera {i}" for i in self.available_cameras])
            self.select_camera(self.available_cameras[0])

        # 创建一个定时器，用于自动更新状态
        self.timer = QTimer(self)
        self.timer.setInterval(1000)  # 设置时间间隔为1000毫秒（1秒）
        self.timer.timeout.connect(self.update_status)
        self.timer.start()

    def initUI(self):
        # 设置窗口的尺寸和标题
        self.setGeometry(200, 200, 1200, 800)
        self.setWindowTitle('Simple PyQt5 App with Multiple Cameras')

        # 使用布局管理器来组织控件
        layout = QVBoxLayout(self)

        # 创建一个用于显示视频的 QLabel
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        layout.addWidget(self.image_label)

        # 创建摄像头选择下拉列表，放在视频正下方
        self.camera_selector = QComboBox(self)
        self.camera_selector.currentIndexChanged.connect(self.on_camera_selected)
        layout.addWidget(self.camera_selector)

        # 创建显示夹爪状态的标签
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

        # 将布局设置为窗口的主布局
        self.setLayout(layout)

    def select_camera(self, port):
        self.thread.select_camera(port)
        self.thread.start()

    def on_camera_selected(self, index):
        if self.available_cameras:
            selected_port = self.available_cameras[index]
            self.select_camera(selected_port)

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
