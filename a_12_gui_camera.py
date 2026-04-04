# pip3 install opencv-contrib-python PyQt5==5.12 qtawesome pyqtgraph pyyaml zmq ur_rtde

from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
#import qtawesome
#import pyqtgraph as pg

import sys, time
import cv2, yaml
import numpy as np

# from pupil_apriltags import Detector
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import rtde_control
import rtde_receive

import sys
#from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import time
import serial
from ControlGripper_DH3 import SetCmd #, ControlRoot, ReadStatus  # 从DH3模块中直接导入需要的类



class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口的尺寸和标题
        self.setGeometry(1200, 400, 1600, 1200)
        self.setWindowTitle('ActiveSPN controller')
        self.setFixedSize(800, 700)  # 固定窗口大小
        self.disply_width = 640  # 显示图像的宽度
        self.display_height = 480  # 显示图像的高度

        # 创建一个按钮，并设置它的位置和点击事件
        btn = QPushButton('Click Me', self)
        btn.move(100, 80)
        btn.clicked.connect(self.show_message)

        # 主窗口布局
        self.main_widget = QWidget()
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        #self.setCentralWidget(self.main_widget)

        self.left_widget = QWidget()
        self.left_widget.setObjectName("left_widget")
        self.left_layout = QGridLayout()
        self.left_widget.setLayout(self.left_layout)

        self.right_widget = QWidget()
       

        self.right_widget.setObjectName("right_widget")
        self.right_layout = QGridLayout()
        self.right_widget.setLayout(self.right_layout)

        self.main_layout.addWidget(self.left_widget, 0,0,10,7)  # 添加左侧布局部件
        # 创建显示图像的标签
        self.left_layout.addWidget(QLabel("Camera Capture"))
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.left_layout.addWidget(self.image_label)



    def show_message(self):
        # 显示消息框
        QMessageBox.information(self, 'Message', 'Hello, PyQt5!')

def main():
    app = QApplication(sys.argv)
    ex = SimpleApp()
    ex.show()
    sys.exit(app.exec_())

    gripper = SetCmd() 




if __name__ == '__main__':
    main()