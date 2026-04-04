from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys, cv2, numpy as np
from ControlGripper_DH3gui import SetCmd
import time
import serial

####DH-3
gripper = SetCmd() 
time.sleep(1)
step = 1

#gripper.Position(60)
gripper.angle(60)
gripper.Force(10)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, port):
        super().__init__()
        self.cap = None
        self.port = port
        self.running = True
        self.mutex = QMutex()

    def run(self):
        self.cap = cv2.VideoCapture(self.port)
        if not self.cap.isOpened():
            print(f"Camera at port {self.port} could not be opened.")
            return

        while self.running:
            self.mutex.lock()
            try:
                ret, cv_img = self.cap.read()
                if ret:
                    self.change_pixmap_signal.emit(cv_img)
            finally:
                self.mutex.unlock()
            QThread.msleep(10)

    def set_exposure(self, value):
        if self.cap is not None and self.cap.isOpened():
            print(f"Setting exposure to {value}")  # 打印尝试设置的曝光值
            success = self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
            if success:
                print(f"Exposure set to {value} successfully")
            else:
                print(f"Failed to set exposure to {value}")

    def stop(self):
        self.mutex.lock()
        try:
            self.running = False
            if self.cap:
                self.cap.release()
        finally:
            self.mutex.unlock()
        self.wait()

class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.camera_ports = self.detect_cameras()
        self.selected_ports = self.select_cameras([5, 2, 3, 4])#1/
        self.labels = ["OUTSIDE", "Inner image LEFT", "Inner image RIGHT", "Inner image BEHIND"]
        self.initUI()

        self.threads = []
        for i, port in enumerate(self.selected_ports):
            thread = VideoThread(port)
            thread.change_pixmap_signal.connect(self.get_update_image_function(i))
            self.threads.append(thread)
            thread.start()

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_status)
        self.timer.start()

    def detect_cameras(self):
        available_ports = []
        for port in range(50):
            cap = cv2.VideoCapture(port)
            if cap.isOpened():
                available_ports.append(port)
                cap.release()
        return available_ports

    def select_cameras(self, indices):
        selected_ports = []
        for i in indices:
            if i < len(self.camera_ports):
                selected_ports.append(self.camera_ports[i])
        return selected_ports

    def initUI(self):
        self.setGeometry(100, 100, 1100, 800)
        self.setWindowTitle('Multi-Camera Display')

        main_layout = QVBoxLayout(self)

        # 夹爪控制和状态显示布局
        control_layout = QVBoxLayout()

        self.force_label = QLabel('Force: ', self)
        self.position_label = QLabel('Position: ', self)
        self.angle_label = QLabel('Angle: ', self)
        self.feedback_label = QLabel('Feedback: ', self)
        self.feedback_label.setWordWrap(True)

        control_layout.addWidget(self.force_label)
        control_layout.addWidget(self.position_label)
        control_layout.addWidget(self.angle_label)
        control_layout.addWidget(self.feedback_label)

        self.position_input = QLineEdit(self)
        self.position_input.setPlaceholderText("Enter Position (0-95)")
        self.angle_input = QLineEdit(self)
        self.angle_input.setPlaceholderText("Enter Angle (0-100)")
        self.force_input = QLineEdit(self)
        self.force_input.setPlaceholderText("Enter Force (10-90)")

        control_layout.addWidget(QLabel('Position:'))
        control_layout.addWidget(self.position_input)
        control_layout.addWidget(QLabel('Angle:'))
        control_layout.addWidget(self.angle_input)
        control_layout.addWidget(QLabel('Force:'))
        control_layout.addWidget(self.force_input)

        move_btn = QPushButton('MOVE', self)
        move_btn.clicked.connect(self.move_gripper)
        control_layout.addWidget(move_btn)

        main_layout.addLayout(control_layout)

        # 摄像头视频布局
        camera_layout = QHBoxLayout()
        self.image_labels = []
        self.exposure_sliders = []  # 曝光滑块列表
        for i in range(len(self.selected_ports)):
            vbox = QVBoxLayout()
            video_label = QLabel(self)
            video_label.setFixedSize(640, 480)
            vbox.addWidget(video_label)
            self.image_labels.append(video_label)

            port_label = QLabel(f"Camera Port: {self.selected_ports[i]}", self)
            port_label.setAlignment(Qt.AlignCenter)
            vbox.addWidget(port_label)

            position_label = QLabel(self.labels[i], self)
            position_label.setAlignment(Qt.AlignCenter)
            vbox.addWidget(position_label)

            # 曝光调整滑块
            exposure_slider = QSlider(Qt.Horizontal)
            exposure_slider.setRange(-10, 10)  # 曝光值范围，根据摄像头调整
            exposure_slider.setValue(0)  # 默认曝光值
            exposure_slider.valueChanged.connect(lambda value, idx=i: self.adjust_exposure(idx, value))
            vbox.addWidget(QLabel("Exposure"))
            vbox.addWidget(exposure_slider)
            self.exposure_sliders.append(exposure_slider)

            camera_layout.addLayout(vbox)

        main_layout.addLayout(camera_layout)

        # 添加姿态输出
        pose_layout = QVBoxLayout()
        self.pose_labels = {
            'x': QLabel('x: ', self),
            'y': QLabel('y: ', self),
            'z': QLabel('z: ', self),
            'roll': QLabel('roll: ', self),
            'pitch': QLabel('pitch: ', self),
            'yaw': QLabel('yaw: ', self)
        }
        for key in self.pose_labels:
            pose_layout.addWidget(self.pose_labels[key])

        main_layout.addLayout(pose_layout)

        self.setLayout(main_layout)

    def adjust_exposure(self, index, value):
        """调整摄像头曝光值"""
        self.threads[index].set_exposure(value)

    def get_update_image_function(self, index):
        def update_image(cv_img):
            qt_img = self.convert_cv_qt(cv_img)
            self.image_labels[index].setPixmap(qt_img)
        return update_image

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        for thread in self.threads:
            thread.stop()
        event.accept()

    def update_status(self):
        force_value = gripper.ForceRead()
        self.force_label.setText(f"Force: {force_value}")

        position_value = gripper.PositionRead()
        self.position_label.setText(f"Position: {position_value}")

        angle_value = gripper.angleRead()
        self.angle_label.setText(f"Angle: {angle_value}")

        feedback_value = gripper.FeedbackRead()
        self.feedback_label.setText(f"Feedback: {feedback_value}")

        # 更新姿态输出
        pose_values = [18.16,108.29,3.71,7.12,5.36,95.81]  # 此处为虚构函数，请根据实际应用中调整
        self.pose_labels['x'].setText(f"x: {pose_values[0]}")
        self.pose_labels['y'].setText(f"y: {pose_values[1]}")
        self.pose_labels['z'].setText(f"z: {pose_values[2]}")
        self.pose_labels['roll'].setText(f"roll: {pose_values[3]}")
        self.pose_labels['pitch'].setText(f"pitch: {pose_values[4]}")
        self.pose_labels['yaw'].setText(f"yaw: {pose_values[5]}")

    def move_gripper(self):
        try:
            position = int(self.position_input.text())
            angle = int(self.angle_input.text())
            force = int(self.force_input.text())

            gripper.Position(position)
            gripper.angle(angle)
            gripper.Force(force)

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