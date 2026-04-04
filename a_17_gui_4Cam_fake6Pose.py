from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, QPushButton, QMessageBox, QGroupBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QMutex
import sys
import cv2
import numpy as np
from ControlGripper_DH3gui import SetCmd
import time

class GripperController:
    def __init__(self):
        self.gripper = SetCmd()
        time.sleep(1)

    def set_position(self, position):
        self.gripper.Position(position)

    def set_angle(self, angle):
        self.gripper.angle(angle)

    def set_force(self, force):
        self.gripper.Force(force)

    def read_status(self):
        return {
            'force': self.gripper.ForceRead(),
            'position': self.gripper.PositionRead(),
            'angle': self.gripper.angleRead(),
            'feedback': self.gripper.FeedbackRead()
        }

class VideoCaptureThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, port, index):
        super().__init__()
        self.port = port
        self.index = index
        self.cap = None
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
                ret, frame = self.cap.read()
                if ret:
                    self.new_frame_signal.emit(frame, self.index)
            finally:
                self.mutex.unlock()
            QThread.msleep(10)

    def set_exposure(self, exposure_value):
        if self.cap and self.cap.isOpened():
            if not self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_value)):
                print(f"Failed to set exposure to {exposure_value}")

    def stop(self):
        self.mutex.lock()
        try:
            self.running = False
            if self.cap:
                self.cap.release()
        finally:
            self.mutex.unlock()
        self.wait()

class CMMMeasurementApp(QWidget):
    def __init__(self):
        super().__init__()
        self.gripper = GripperController()
        self.camera_ports = self.detect_available_cameras()
        self.selected_ports = self.select_cameras([5, 2, 3, 4])
        self.camera_labels = ["OUTSIDE", "Inner LEFT", "Inner RIGHT", "Inner BEHIND"]
        self.video_threads = []

        # 预设pose值
        self.pose_values = [18.16, 108.29, 3.71, 7.12, 5.36, 5.81]

        self.init_ui()
        self.init_video_threads()
        self.start_status_update_timer()

    def detect_available_cameras(self):
        available_ports = []
        for port in range(50):
            cap = cv2.VideoCapture(port)
            if cap.isOpened():
                available_ports.append(port)
                cap.release()
        return available_ports

    def select_cameras(self, indices):
        return [self.camera_ports[i] for i in indices if i < len(self.camera_ports)]

    def init_ui(self):
        self.setWindowTitle("ActiveSPN")
        self.setGeometry(100, 100, 1100, 800)
        self.setStyleSheet("background-color: #f0f0f0;")

        main_layout = QVBoxLayout(self)

        # 夹爪控制布局
        self.setup_gripper_control_ui(main_layout)

        # 摄像头显示布局
        self.setup_camera_display_ui(main_layout)

        # 姿态信息显示布局
        self.setup_pose_display_ui(main_layout)

    def setup_gripper_control_ui(self, layout):
        control_group = QGroupBox("Gripper Parameters")
        control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4A90E2;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
                background-color: #FFFFFF;
            }
            QLabel {
                font-size: 14px;
            }
            QLineEdit, QPushButton {
                font-size: 14px;
                padding: 5px;
                margin: 5px;
            }
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)

        control_layout = QVBoxLayout()
        self.force_label = QLabel("Force: ", self)
        self.position_label = QLabel("Position: ", self)
        self.angle_label = QLabel("Angle: ", self)
        self.feedback_label = QLabel("Feedback: ", self)

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

        control_layout.addWidget(self.position_input)
        control_layout.addWidget(self.angle_input)
        control_layout.addWidget(self.force_input)

        move_btn = QPushButton("MOVE", self)
        move_btn.clicked.connect(self.move_gripper)
        control_layout.addWidget(move_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

    def setup_camera_display_ui(self, layout):
        camera_group = QGroupBox("Inner Vision")
        camera_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4A90E2;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
                background-color: #FFFFFF;
            }
            QLabel {
                font-size: 14px;
            }
            QSlider {
                margin: 5px;
            }
        """)

        camera_layout = QHBoxLayout()
        self.image_labels = []

        for i, port in enumerate(self.selected_ports):
            vbox = QVBoxLayout()
            video_label = QLabel(self)
            video_label.setFixedSize(320, 240)
            video_label.setStyleSheet("border: 1px solid #ddd;")
            self.image_labels.append(video_label)
            vbox.addWidget(video_label)
            vbox.addWidget(QLabel(f"Camera Port: {port}", self))
            vbox.addWidget(QLabel(self.camera_labels[i], self))

            exposure_slider = QSlider(Qt.Horizontal)
            exposure_slider.setRange(-10, 10)
            exposure_slider.valueChanged.connect(lambda value, idx=i: self.adjust_camera_exposure(idx, value))
            vbox.addWidget(QLabel("Exposure"))
            vbox.addWidget(exposure_slider)

            camera_layout.addLayout(vbox)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

    def setup_pose_display_ui(self, layout):
        pose_group = QGroupBox("Pose Estimation")
        pose_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4A90E2;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
                background-color: #FFFFFF;
            }
            QLabel {
                font-size: 14px;
                padding: 2px;
            }
        """)

        pose_layout = QVBoxLayout()
        self.pose_labels = {key: QLabel(f"{key}: ", self) for key in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}
        for label in self.pose_labels.values():
            pose_layout.addWidget(label)

        pose_group.setLayout(pose_layout)
        layout.addWidget(pose_group)

    def init_video_threads(self):
        for i, port in enumerate(self.selected_ports):
            thread = VideoCaptureThread(port, i)
            thread.new_frame_signal.connect(self.update_video_frame)
            self.video_threads.append(thread)
            thread.start()

    def adjust_camera_exposure(self, index, value):
        self.video_threads[index].set_exposure(value)

    def update_video_frame(self, frame, index):
        qt_img = self.convert_cv_qt(frame)
        self.image_labels[index].setPixmap(qt_img)

    def convert_cv_qt(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img.scaled(320, 240, Qt.KeepAspectRatio))

    def move_gripper(self):
        try:
            position = int(self.position_input.text())
            angle = int(self.angle_input.text())
            force = int(self.force_input.text())
            self.gripper.set_position(position)
            self.gripper.set_angle(angle)
            self.gripper.set_force(force)
            self.update_gripper_status()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers.")

    def start_status_update_timer(self):
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_gripper_status)
        self.timer.start()

    def update_gripper_status(self):
        status = self.gripper.read_status()
        self.force_label.setText(f"Force: {status['force']}")
        self.position_label.setText(f"Position: {status['position']}")
        self.angle_label.setText(f"Angle: {status['angle']}")
        self.feedback_label.setText(f"Feedback: {status['feedback']}")

        # 更新姿态输出
        self.pose_labels['x'].setText(f"x: {self.pose_values[0]}")
        self.pose_labels['y'].setText(f"y: {self.pose_values[1]}")
        self.pose_labels['z'].setText(f"z: {self.pose_values[2]}")
        self.pose_labels['roll'].setText(f"roll: {self.pose_values[3]}")
        self.pose_labels['pitch'].setText(f"pitch: {self.pose_values[4]}")
        self.pose_labels['yaw'].setText(f"yaw: {self.pose_values[5]}")

    def closeEvent(self, event):
        for thread in self.video_threads:
            thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    main_window = CMMMeasurementApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()