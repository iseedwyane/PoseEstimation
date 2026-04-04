# pip3 install opencv-contrib-python PyQt5==5.12 qtawesome pyqtgraph pyyaml zmq ur_rtde

from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qtawesome
import pyqtgraph as pg

import sys, time
import cv2, yaml
import numpy as np

# from pupil_apriltags import Detector
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import rtde_control
import rtde_receive

class RobotThread(QThread):
    change_tcp_signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
        self.tcp_pose_vecs = np.array([[0,0,0,0,0,0]])
        self.force = 0
    
    def run(self):
        while True:
            pose = np.array( self.rtde_r.getActualTCPPose() )
            self.force = np.array( self.rtde_r.getActualTCPForce() )
            self.tcp_pose_vecs = np.concatenate((self.tcp_pose_vecs, pose.reshape([1,6])))
            self.change_tcp_signal.emit(pose)


import cv2.aruco as aruco
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_pose_signal = pyqtSignal(np.ndarray)
    def __init__(self, source='0', config='default'):
        super().__init__()
        self.source = source
        self.config = config
        self._run_flag = True
        if self.config == 'default':
            self.camera_matrix = np.matrix([[304.67164266,   0.        , 336.98629633],
                                            [  0.        , 303.77152712, 205.85572583],
                                            [  0.        ,   0.        ,   1.        ]])
            self.camera_dist = np.matrix([-0.32187833,  0.16486379, -0.00041955, -0.00082774, -0.06473362])
            self.marker_length = 0.005
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            self.parameters = aruco.DetectorParameters_create()
            self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

        else:
            self.camera_matrix = np.matrix(self.config['camera_matrix'])
            self.camera_dist = np.matrix(self.config['camera_dist'])
            self.marker_length = self.config['marker_length']
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            self.parameters = aruco.DetectorParameters_create()
            self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

        self.tag_pose_vecs = np.array([[0,0,0,0,0,0]])

        self.detection_failure = 0
        self.time_cost = []

    def run(self):
        # capture from web cam
        self._run_flag = True
        cap = cv2.VideoCapture(int(self.source))
        cap.set(3,int(self.config['width'])) #设置分辨率
        cap.set(4,int(self.config['height']))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # fixed exposure
        cap.set(cv2.CAP_PROP_EXPOSURE,2)

        while self._run_flag:
            
            ret, color_image = cap.read()
            if ret is False:
                # print("****Fail to read image from camera!")
                # self._run_flag = False
                raise ValueError('Fail to read image from camera %s, please choose another camera'%self.source)
            else:
                s = time.time()
                gray_ = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((5,5),np.float32)/25
                gray = cv2.filter2D(gray_,-1,kernel)
                current_corners, current_ids, _ = aruco.detectMarkers(gray, 
                                                self.aruco_dict, 
                                                parameters=self.parameters)
                if current_ids is None:
                    self.detection_failure += 1
                    self.change_pixmap_signal.emit(color_image)
                    continue
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(current_corners,
                                                                  self.marker_length,
                                                                  self.camera_matrix,
                                                                  self.camera_dist)

                rr = R.from_rotvec(rvecs[0])
                rpy = rr.as_euler('zyx', degrees=True)[0][::-1]
                # rpy = rr.as_euler('xyz', degrees=True)[0] # This is the correct one 

                # print('rpy:', rpy)
                self.time_cost.append(time.time()-s)

                pose = np.concatenate((tvecs[0][0] * 1000, rpy))
                self.tag_pose_vecs = np.concatenate((self.tag_pose_vecs, pose.reshape([1,6])))
                color_image_copy = np.copy(color_image)
                for i in range(len(rvecs)):
                    color_image_result = aruco.drawAxis(color_image_copy, self.camera_matrix, self.camera_dist, rvecs[i], tvecs[i], self.marker_length)
                # cv2.imwrite("raw.png",color_image)
                # cv2.imwrite("detection.png",color_image_copy)

                self.change_pixmap_signal.emit(color_image_result)
                self.change_pose_signal.emit(pose)
                # print("------------------wanfang:",tvecs[0] * 1000, rpy, pose)
        # shut down capture system
        print("Shut down the connection to camera %s."%self.source)
        cap.release()
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


# class DetectionThread(QThread)
MAX_SPEED = 0.2
MAX_ROTATION = 0.7
class App(QMainWindow):
    def __init__(self, robot_connection = False): # False, read, control
        super().__init__()
        self.config = yaml.load(open('controller_config.yaml'), Loader=yaml.FullLoader)
        self.tcp_force_vecs = np.array([[0,0,0,0,0,0]])
        self.robot_connection = robot_connection
        self.setWindowTitle("AprilTag controller")
        self.setFixedSize(800, 700)
        self.disply_width = 640
        self.display_height = 480

        # setup for robot controller
        if robot_connection == 'control':
            self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
        self.acceleration = 2
        self.controllers = ["rotation-position", "rotation-rotation", "position-position", "position-xy","position-rotation"]
        self.controller_selected = self.controllers[3]

        # main window
        self.main_widget = QWidget()
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.left_widget = QWidget()
        self.left_widget.setObjectName("left_widget")
        self.left_layout = QGridLayout()
        self.left_widget.setLayout(self.left_layout)

        self.right_widget = QWidget()
        self.right_widget.setObjectName("right_widget")
        self.right_layout = QGridLayout()
        self.right_widget.setLayout(self.right_layout)

        self.main_layout.addWidget(self.left_widget, 0,0,10,7)
        
        # create the label that holds the image
        self.left_layout.addWidget(QLabel("Camera Capture"))
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.left_layout.addWidget(self.image_label)

        self.playconsole_widget = QWidget()
        self.playconsole_layout = QGridLayout()
        self.playconsole_widget.setLayout(self.playconsole_layout)
        self.play_button = QPushButton(qtawesome.icon('fa.play', color="#F76677", font=18), "")
        self.play_button.setIconSize(QSize(30,30))
        self.play_button.clicked.connect(self.start_camera_streaming)
        self.pause_button = QPushButton(qtawesome.icon('fa.pause', color="#F76677", font=18), "")
        self.pause_button.setIconSize(QSize(30,30))
        self.pause_button.clicked.connect(self.pause_camera_streaming)
        self.playconsole_layout.addWidget(self.play_button,0,0)
        self.playconsole_layout.addWidget(self.pause_button,0,1)
        self.left_layout.addWidget(self.playconsole_widget)

        # create speed controller
        self.speed_widget = QWidget()
        self.speed_layout = QHBoxLayout()
        self.speed_widget.setLayout(self.speed_layout)
        self.l1 = QLabel("Speed")
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(0)
        self.sl.setMaximum(100)
        self.sl.setValue(50)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.valueChanged.connect(self.change_speed)
        self.speed_layout.addWidget(self.l1)
        self.speed_layout.addWidget(self.sl)
        self.left_layout.addWidget(self.speed_widget)
        self.speed_magnitude = MAX_SPEED * 50/100
        self.rotation_magnitude = MAX_ROTATION * 50/100

        # create pose drawing

        # create april-tag pose display
        self.tag_pose_widget = QWidget()
        self.tag_pose_layout = QGridLayout()
        self.tag_pose_widget.setLayout(self.tag_pose_layout)
        self.tag_pose_layout.addWidget(QLabel("Detect tag pose:"),0,0,1,2)
        self.tag_pose_layout.addWidget(QLabel("X:"),1,0)
        self.tag_pose_layout.addWidget(QLabel("Y:"),2,0)
        self.tag_pose_layout.addWidget(QLabel("Z:"),3,0)
        self.tag_pose_layout.addWidget(QLabel("R:"),4,0)
        self.tag_pose_layout.addWidget(QLabel("P:"),5,0)
        self.tag_pose_layout.addWidget(QLabel("Y:"),6,0)
        self.tag_pose_labels = []
        for i in range(6):
            self.tag_pose_labels.append(QLabel())
            self.tag_pose_layout.addWidget(self.tag_pose_labels[-1], i+1,1)
        for i in range(3):
            self.tag_pose_layout.addWidget(QLabel("mm"), i+1,2)
        for i in range(3,6):
            self.tag_pose_layout.addWidget(QLabel("deg"), i+1,2)
        self.main_layout.addWidget(self.tag_pose_widget, 0,7,5,3)

        # create robot TCP pose display
        self.TCP_widget = QWidget()
        self.TCP_layout = QGridLayout()
        self.TCP_widget.setLayout(self.TCP_layout)
        self.TCP_layout.addWidget(QLabel("Robot TCP:"),0,0,1,2)
        self.TCP_layout.addWidget(QLabel("X:"),1,0)
        self.TCP_layout.addWidget(QLabel("Y:"),2,0)
        self.TCP_layout.addWidget(QLabel("Z:"),3,0)
        self.TCP_layout.addWidget(QLabel("RX:"),4,0)
        self.TCP_layout.addWidget(QLabel("RY:"),5,0)
        self.TCP_layout.addWidget(QLabel("RZ:"),6,0)
        self.tcp_pose_labels = []
        for i in range(6):
            self.tcp_pose_labels.append(QLabel())
            self.TCP_layout.addWidget(self.tcp_pose_labels[-1], i+1,1)
        for i in range(3):
            self.TCP_layout.addWidget(QLabel("mm"), i+1,2)
        for i in range(3,6):
            self.TCP_layout.addWidget(QLabel("rad"), i+1,2)
        self.main_layout.addWidget(self.TCP_widget, 5,7,5,3)

        # create toolbar on the top
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        toolbar.addWidget(QLabel("Camera:"))
        self.cameraList = QComboBox()
        toolbar.addWidget(self.cameraList)
        for i in range(3):
            self.cameraList.addItem(str(i))
        self.cameraList.setCurrentText(str(0))
        self.cameraList.currentTextChanged.connect(self.change_camera)

        toolbar.addWidget(QLabel("     Controller:"))
        self.controllerList = QComboBox()
        toolbar.addWidget(self.controllerList)
        for i in range(len(self.controllers)):
            self.controllerList.addItem(self.controllers[i])
        self.controllerList.setCurrentText(self.controller_selected)
        self.controllerList.currentTextChanged.connect(self.change_controller)

        # create the video capture thread
        self.video_thread = VideoThread(config = self.config['vision'])
        # connect its signal to the update_image slot
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.change_pose_signal.connect(self.update_tag_pose)
        if self.robot_connection is not False:
            self.video_thread.change_pose_signal.connect(self.move_robot)
        self.video_thread.start()

        if self.robot_connection is not False:
            self.robot_thread = RobotThread()
            self.robot_thread.change_tcp_signal.connect(self.update_tcp_pose)        
            self.robot_thread.start()

    def start_camera_streaming(self):
        self.video_thread.start()

    def pause_camera_streaming(self):
        self.video_thread.stop()

    def change_camera(self, source):
        self.video_thread.stop()
        self.video_thread = VideoThread(source, config = self.config['vision'])
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.change_pose_signal.connect(self.update_tag_pose)
        if self.robot_connection is not False:
            self.video_thread.change_pose_signal.connect(self.move_robot)
        try:
            self.video_thread.start()
        except ValueError as error:
            print("Fail to connect to the source camera, please choose another camera")

    def closeEvent(self, event):
        if self.video_thread._run_flag == True:
            np.save('tag_pose_vecs.npy',self.video_thread.tag_pose_vecs[1:,:])
            np.save('time_cost.npy',self.video_thread.time_cost)
            np.save('tcp_pose_vecs.npy',self.robot_thread.tcp_pose_vecs)
            np.save('tcp_force_vecs.npy',self.tcp_force_vecs[1:,:])

            print("detection_failure",self.video_thread.detection_failure, 'success:', self.video_thread.tag_pose_vecs.shape[0])
            self.video_thread.stop()
        if self.robot_connection == 'control':
            self.rtde_c.stopScript()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def update_tag_pose(self, tag_pose):
        if self.robot_connection is not False:
            self.tcp_force_vecs = np.concatenate((self.tcp_force_vecs, self.robot_thread.force.reshape([1,6])))
            print(self.robot_thread.force.reshape([1,6]))
        for i in range(6):
            self.tag_pose_labels[i].setText('%.1f'%(tag_pose[i]))

    def update_tcp_pose(self, tcp_pose):
        for i in range(3):
            self.tcp_pose_labels[i].setText('%.1f'%(tcp_pose[i]*1000))
        for i in range(3,6):
            self.tcp_pose_labels[i].setText('%.1f'%(tcp_pose[i]))
    
    def move_robot(self, tag_pose):
        s = time.time()
        speed_vector = self.get_speed()
        if speed_vector is not None and self.robot_connection == 'control':
            self.rtde_c.speedL(speed_vector, self.acceleration, 0.03)
            # print("Speed: ",speed_vector, time.time()-s)

    def get_speed(self):
        if self.video_thread.tag_pose_vecs.shape[0] < 100:
            return None
        # todo: smooth
        win_size = 5
        temp_r0 = R.from_euler('zyx', self.video_thread.tag_pose_vecs[50:100, 5:2:-1], degrees=True)
        rpy0 = temp_r0.mean().as_euler('zyx', degrees=True)[::-1]
        p0 = np.concatenate((np.mean(self.video_thread.tag_pose_vecs[50:100, 0:3], axis=0), rpy0))
        
        temp_rt = R.from_euler('zyx', self.video_thread.tag_pose_vecs[-win_size:, 5:2:-1], degrees=True)
        rpy = temp_rt.mean().as_euler('zyx', degrees=True)[::-1]
        pt = np.concatenate((np.mean(self.video_thread.tag_pose_vecs[-win_size:, 0:3], axis=0), rpy))
        
        dp = pt - p0
        # print('original dp: ', dp)
        
        # Todo: tcp rotation with reference to base
        robot_tcp_rt = np.array([float(self.tcp_pose_labels[0].text()), float(self.tcp_pose_labels[1].text()), float(self.tcp_pose_labels[2].text()),
               float(self.tcp_pose_labels[3].text()), float(self.tcp_pose_labels[4].text()), float(self.tcp_pose_labels[5].text())])
        bMe = (R.from_rotvec(robot_tcp_rt[3:6])).as_matrix()
        
        # camera pose with reference to tool flange
        eMc = R.from_euler('Z', 90, degrees=True).as_matrix()
        # calculate the camera pose with reference to robot base
        bMc = np.dot(bMe, eMc)
        dp_translation = np.dot(bMc, dp[0:3])
        
        # calculate tag pose difference with reference to robot base
        init_pose = R.from_euler('zyx', rpy0[::-1], degrees=True)
        init_bMtag = np.dot(bMc, init_pose.as_matrix())
        init_euler = (R.from_matrix(init_bMtag)).as_euler('zyx', degrees=True)[::-1]
        
        current_pose = R.from_euler('zyx', rpy[::-1], degrees=True)
        current_bMtag = np.dot(bMc, current_pose.as_matrix())
        current_euler = (R.from_matrix(current_bMtag)).as_euler('zyx', degrees=True)[::-1]
        
        dp_rotation = current_euler - init_euler
        
        dp = np.concatenate((dp_translation, dp_rotation))
        # print('new dp: ', dp)

        if self.controller_selected == "rotation-position":
            # roll is rotation around x-axis, the human intuition is move back and forth along y-axis
            # pitch is rotation around y-axis, the human intuition is move back and forth along x-axis. The oriention
            roll, pitch, yaw = (dp[3:] >= 10).astype("float") + \
                    (np.logical_and(dp[3:] > 3, dp[3:] < 10)).astype("float")*(dp[3:]-3)/7 \
                    - (dp[3:] <= -10).astype("float") \
                    + (np.logical_and(dp[3:] < -3, dp[3:] > -10)).astype("float")*(dp[3:]+3)/7
            speed_vector = np.concatenate( ( self.speed_magnitude * np.array([pitch, -roll, yaw]), [0,0,0] ) )

        elif self.controller_selected == "rotation-rotation":
            scale = (dp[3:] >= 10).astype("float") + \
                    (np.logical_and(dp[3:] > 3, dp[3:] < 10)).astype("float")*(dp[3:]-3)/7 \
                    - (dp[3:] <= -10).astype("float") \
                    + (np.logical_and(dp[3:] < -3, dp[3:] > -10)).astype("float")*(dp[3:]+3)/7
            speed_vector = np.concatenate( ( [0,0,0], self.speed_magnitude * np.array(scale) ) )
        
        elif self.controller_selected == "position-position":
            # xy movement corresponding to xy movement, yaw -> z movement
            dx, dy = (dp[:2] >= 5).astype("float") + \
                    (np.logical_and(dp[:2] > 0.5, dp[:2] < 5)).astype("float")*(dp[:2]-0.5)/4.5 \
                    - (dp[:2] <= -5).astype("float") \
                    + (np.logical_and(dp[:2] < -0.5, dp[:2] > -5)).astype("float")*(dp[:2]+0.5)/4.5
            drz = (dp[5] >= 10).astype("float") + \
                    (np.logical_and(dp[5] > 1, dp[5] < 10)).astype("float")*(dp[5]-1)/9 \
                    - (dp[5] <= -10).astype("float") \
                    + (np.logical_and(dp[5] < -1, dp[5] > -10)).astype("float")*(dp[5]+1)/9
            speed_vector = np.concatenate((self.speed_magnitude * np.array([dx, dy, drz]), [0, 0, 0]))

        elif self.controller_selected == "position-xy":
            # xy movement corresponding to xy movement, yaw -> z movement
            dx, dy = (dp[:2] >= 5).astype("float") + \
                    (np.logical_and(dp[:2] > 0.5, dp[:2] < 5)).astype("float")*(dp[:2]-0.5)/4.5 \
                    - (dp[:2] <= -5).astype("float") \
                    + (np.logical_and(dp[:2] < -0.5, dp[:2] > -5)).astype("float")*(dp[:2]+0.5)/4.5
            speed_vector = np.concatenate((self.speed_magnitude * np.array([dx, dy, 0]), [0, 0, 0]))

        elif self.controller_selected == "position-rotation":
            # xy movement corresponding to xy movement, yaw -> z movement
            dx, dy = (dp[:2] >= 5).astype("float") + \
                    (np.logical_and(dp[:2] > 0.5, dp[:2] < 5)).astype("float")*(dp[:2]-0.5)/4.5 \
                    - (dp[:2] <= -5).astype("float") \
                    + (np.logical_and(dp[:2] < -0.5, dp[:2] > -5)).astype("float")*(dp[:2]+0.5)/4.5
            drz = (dp[5] >= 10).astype("float") + \
                    (np.logical_and(dp[5] > 1, dp[5] < 10)).astype("float")*(dp[5]-1)/9 \
                    - (dp[5] <= -10).astype("float") \
                    + (np.logical_and(dp[5] < -1, dp[5] > -10)).astype("float")*(dp[5]+1)/9
            speed_vector = np.concatenate( ( [0,0,0], self.rotation_magnitude * np.array([-dy, dx, -drz]) ) )
        
        else:
            speed_vector = np.concatenate( ( self.speed_magnitude*( (tvec > 5).astype("float") - (rpy < -5).astype("float") ), np.array([0,0,0]) ) )
        
        return speed_vector

    def change_speed(self, percent):
        self.speed_magnitude = MAX_SPEED * percent/100
        self.rotation_magnitude = MAX_ROTATION * percent/100

    def change_controller(self, controller):
        self.controller_selected = controller

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())














































```python
# 导入必要的库
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qtawesome
import pyqtgraph as pg

import sys, time
import cv2, yaml
import numpy as np

# 导入Aruco标记库和其他科学计算库
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import rtde_control
import rtde_receive

# 机器人线程类，继承自QThread，用于处理与机器人控制器的实时通信
class RobotThread(QThread):
    change_tcp_signal = pyqtSignal(np.ndarray)  # 用于发射TCP位置更新信号
    def __init__(self):
        super().__init__()
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")  # 初始化机器人接收接口
        self.tcp_pose_vecs = np.array([[0,0,0,0,0,0]])  # 初始化TCP姿态向量
        self.force = 0  # 初始化力值

    # 线程的运行函数，持续获取机器人TCP的位姿和力值
    def run(self):
        while True:
            pose = np.array(self.rtde_r.getActualTCPPose())  # 获取机器人当前TCP位姿
            self.force = np.array(self.rtde_r.getActualTCPForce())  # 获取机器人当前TCP力值
            self.tcp_pose_vecs = np.concatenate((self.tcp_pose_vecs, pose.reshape([1,6])))  # 更新位姿向量
            self.change_tcp_signal.emit(pose)  # 发射位姿更新信号

# 视频线程类，继承自QThread，用于处理摄像头视频流和Aruco标记检测
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)  # 用于发射更新图像信号
    change_pose_signal = pyqtSignal(np.ndarray)  # 用于发射标记位姿更新信号
    def __init__(self, source='0', config='default'):
        super().__init__()
        self.source = source  # 摄像头来源
        self.config = config  # 摄像头配置
        self._run_flag = True  # 线程运行标志

        # 默认配置下的相机参数和Aruco标记参数
        if self.config == 'default':
            self.camera_matrix = np.matrix([[304.67164266,   0.        , 336.98629633],
                                            [  0.        , 303.77152712, 205.85572583],
                                            [  0.        ,   0.        ,   1.        ]])
            self.camera_dist = np.matrix([-0.32187833,  0.16486379, -0.00041955, -0.00082774, -0.06473362])
            self.marker_length = 0.005  # 标记的实际长度
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)  # Aruco标记字典
            self.parameters = aruco.DetectorParameters_create()  # Aruco检测参数
            self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR  # 棱角细化方法

        else:
            # 自定义配置的相机参数和Aruco标记参数
            self.camera_matrix = np.matrix(self.config['camera_matrix'])
            self.camera_dist = np.matrix(self.config['camera_dist'])
            self.marker_length = self.config['marker_length']
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            self.parameters = aruco.DetectorParameters_create()
            self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

        self.tag_pose_vecs = np.array([[0,0,0,0,0,0]])  # 初始化标记位姿向量

        self.detection_failure = 0  # 初始化检测失败计数
        self.time_cost = []  # 初始化时间消耗记录

    # 线程的运行函数，捕捉摄像头视频流并进行Aruco标记检测
    def run(self):
        # 从摄像头捕捉视频流
        self._run_flag = True
        cap = cv2.VideoCapture(int(self.source))
        cap.set(3,int(self.config['width']))  # 设置视频宽度
        cap.set(4,int(self.config['height']))  # 设置视频高度
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 固定曝光
        cap.set(cv2.CAP_PROP_EXPOSURE,2)  # 设置曝光值

        while self._run_flag:
            ret, color_image = cap.read()  # 读取视频帧
            if not ret:
                # 如果无法读取图像，则抛出错误
                raise ValueError('Fail to read image from camera %s, please choose another camera'%self.source)
            else:
                s = time.time()  # 记录开始时间
                gray_ = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
                kernel = np.ones((5,5),np.float32)/25
                gray = cv2.filter2D(gray_,-1,kernel)  # 应用滤波器
                current_corners, current_ids, _ = aruco.detectMarkers(gray, 
                                                self.aruco_dict, 
                                                parameters=self.parameters)  # 检测Aruco标记

                if current_ids is None:
                    self.detection_failure += 1  # 更新检测失败计数
                    self.change_pixmap_signal.emit(color_image)  # 更新显示的图像
                    continue

                # 估计标记的位姿
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(current_corners,
                                                                  self.marker_length,
                                                                  self.camera_matrix,
                                                                  self.camera_dist)

                rr = R.from_rotvec(rvecs[0])  # 从旋转向量构建旋转矩阵
                rpy = rr.as_euler('zyx', degrees=True)[0][::-1]  # 获取欧拉角并翻转顺序

                pose = np.concatenate((tvecs[0][0] * 1000, rpy))  # 组合平移和旋转数据
                self.tag_pose_vecs = np.concatenate((self.tag_pose_vecs, pose.reshape([1,6])))  # 更新标记位姿向量
                color_image_copy = np.copy(color_image)  # 复制原始图像
                for i in range(len(rvecs)):
                    # 在图像上绘制Aruco标记的坐标轴
                    color_image_result = aruco.drawAxis(color_image_copy, self.camera_matrix, self.camera_dist, rvecs[i], tvecs[i], self.marker_length)

                self.change_pixmap_signal.emit(color_image_result)  # 更新显示的图像
                self.change_pose_signal.emit(pose)  # 发射标记位姿更新信号

        # 关闭摄像头连接
        print("Shut down the connection to camera %s."%self.source)
        cap.release()

    # 停止摄像头捕捉线程
    def stop(self):
        self._run_flag = False
        self.wait()

# 主程序类，继承自QMainWindow，用于创建主窗口和用户界面
class App(QMainWindow):
    def __init__(self, robot_connection = False):  # robot_connection参数用于决定是否连接机器人
        super().__init__()
        self.config = yaml.load(open('controller_config.yaml'), Loader=yaml.FullLoader)  # 加载控制器配置文件
        self.tcp_force_vecs = np.array([[0,0,0,0,0,0]])  # 初始化TCP力值向量
        self.robot_connection = robot_connection  # 是否连接机器人的标志
        self.setWindowTitle("AprilTag controller")  # 设置窗口标题
        self.setFixedSize(800, 700)  # 固定窗口大小
        self.disply_width = 640  # 显示图像的宽度
        self.display_height = 480  # 显示图像的高度

        # 根据robot_connection参数设置机器人控制接口
        if robot_connection == 'control':
            self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.10")
        self.acceleration = 2  # 设置加速度
        self.controllers = ["rotation-position", "rotation-rotation", "position-position", "position-xy","position-rotation"]  # 控制器模式列表
        self.controller_selected = self.controllers[3]  # 默认选择的控制器模式

        # 主窗口布局
        self.main_widget = QWidget()
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

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

        # 创建播放和暂停按钮
        self.playconsole_widget = QWidget()
        self.playconsole_layout = QGridLayout()
        self.playconsole_widget.setLayout(self.playconsole_layout)
        self.play_button = QPushButton(qtawesome.icon('fa.play', color="#F76677", font=18), "")
        self.play_button.setIconSize(QSize(30,30))
        self.play_button.clicked.connect(self.start_camera_streaming)
        self.pause_button = QPushButton(qtawesome.icon('fa.pause', color="#F76677", font=18), "")
        self.pause_button.setIconSize(QSize(30,30))
        self.pause_button.clicked.connect(self.pause_camera_streaming)
        self.playconsole_layout.addWidget(self.play_button,0,0)
        self.playconsole_layout.addWidget(self.pause_button,0,1)
        self.left_layout.addWidget(self.playconsole_widget)

        # 创建速度控制器
        self.speed_widget = QWidget()
        self.speed_layout = QHBoxLayout()
        self.speed_widget.setLayout(self.speed_layout)
        self.l1 = QLabel("Speed")
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(0)
        self.sl.setMaximum(100)
        self.sl.setValue(50)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.valueChanged.connect(self.change_speed)
        self.speed_layout.addWidget(self.l1)
        self.speed_layout.addWidget(self.sl)
        self.left_layout.addWidget(self.speed_widget)
        self.speed_magnitude = MAX_SPEED * 50/100  # 计算初始速度
        self.rotation_magnitude = MAX_ROTATION * 50/100  # 计算初始旋转速度

        # 创建Aruco标记位姿显示部件
        self.tag_pose_widget = QWidget()
        self.tag_pose_layout = QGridLayout()
        self.tag_pose_widget.setLayout(self.tag_pose_layout)
        self.tag_pose_layout.addWidget(QLabel("Detect tag pose:"),0,0,1,2)
        self.tag_pose_layout.addWidget(QLabel("X:"),1,0)
        self.tag_pose_layout.addWidget(QLabel("Y:"),2,0)
        self.tag_pose_layout.addWidget(QLabel("Z:"),3,0)
        self.tag_pose_layout.addWidget(QLabel("R:"),4,0)
        self.tag_pose_layout.addWidget(QLabel("P:"),5,0)
        self.tag_pose_layout.addWidget(QLabel("Y:"),6,0)
        self.tag_pose_labels = []
        for i in range(6):
            self.tag_pose_labels.append(QLabel())
            self.tag_pose_layout.addWidget(self.tag_pose_labels[-1], i+1,1)
        for i in range(3):
            self.tag_pose_layout.addWidget(QLabel("mm"), i+1,2)
        for i in range(3,6):
            self.tag_pose_layout.addWidget(QLabel("deg"), i+1,2)
        self.main_layout.addWidget(self.tag_pose_widget, 0,7,5,3)

        # 创建机器人TCP位姿显示部件
        self.TCP_widget = QWidget()
        self.TCP_layout = QGridLayout()
        self.TCP_widget.setLayout(self.TCP_layout)
        self.TCP_layout.addWidget(QLabel("Robot TCP:"),0,0,1,2)
        self.TCP_layout.addWidget(QLabel("X:"),1,0)
        self.TCP_layout.addWidget(QLabel("Y:"),2,0)
        self.TCP_layout.addWidget(QLabel("Z:"),3,0)
        self.TCP_layout.addWidget(QLabel("RX:"),4,0)
        self.TCP_layout.addWidget(QLabel("RY:"),5,0)
        self.TCP_layout.addWidget(QLabel("RZ:"),6,0)
        self.tcp_pose_labels = []
        for i in range(6):
            self.tcp_pose_labels.append(QLabel())
            self.TCP_layout.addWidget(self.tcp_pose_labels[-1], i+1,1)
        for i in range(3):
            self.TCP_layout.addWidget(QLabel("mm"), i+1,2)
        for i in range(3,6):
            self.TCP_layout.addWidget(QLabel("rad"), i+1,2)
        self.main_layout.addWidget(self.TCP_widget, 5,7,5,3)

        # 创建工具栏
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        toolbar.addWidget(QLabel("Camera:"))
        self.cameraList = QComboBox()
        toolbar.addWidget(self.cameraList)
        for i in range(3):
            self.cameraList.addItem(str(i))
        self.cameraList.setCurrentText(str(0))
        self.cameraList.currentTextChanged.connect(self.change_camera)

        toolbar.addWidget(QLabel("     Controller:"))
        self.controllerList = QComboBox()
        toolbar.addWidget(self.controllerList)
        for i in range(len(self.controllers)):
            self.controllerList.addItem(self.controllers[i])
        self.controllerList.setCurrentText(self.controller_selected)
        self.controllerList.currentTextChanged.connect(self.change_controller)

        # 创建视频捕捉线程
        self.video_thread = VideoThread(config = self.config['vision'])
        self.video_thread.change_pixmap_signal.connect(self.update_image)  # 连接图像更新信号
        self.video_thread.change_pose_signal.connect(self.update_tag_pose)  # 连接标记位姿更新信号
        if self.robot_connection is not False:
            self.video_thread.change_pose_signal.connect(self.move_robot)  # 连接机器人控制信号
        self.video_thread.start()

        if self.robot_connection is not False:
            self.robot_thread = RobotThread()
            self.robot_thread.change_tcp_signal.connect(self.update_tcp_pose)  # 连接TCP位姿更新信号
            self.robot_thread.start()

    # 启动摄像头视频流
    def start_camera_streaming(self):
        self.video_thread.start()

    # 暂停摄像头视频流
    def pause_camera_streaming(self):
        self.video_thread.stop()

    # 切换摄像头
    def change_camera(self, source):
        self.video_thread.stop()
        self.video_thread = VideoThread(source, config = self.config['vision'])
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.change_pose_signal.connect(self.update_tag_pose)
        if self.robot_connection is not False:
            self.video_thread.change_pose_signal.connect(self.move_robot)
        try:
            self.video_thread.start()
        except ValueError as error:
            print("Fail to connect to the source camera, please choose another camera")

    # 关闭事件处理函数
    def closeEvent(self, event):
        if self.video_thread._run_flag == True:
            np.save('tag_pose_vecs.npy',self.video_thread.tag_pose_vecs[1:,:])  # 保存标记位姿数据
            np.save('time_cost.npy',self.video_thread.time_cost)  # 保存时间消耗数据
            np.save('tcp_pose_vecs.npy',self.robot_thread.tcp_pose_vecs)  # 保存TCP位姿数据
            np.save('tcp_force_vecs.npy',self.tcp_force_vecs[1:,:])  # 保存TCP力值数据

            print("detection_failure",self.video_thread.detection_failure, 'success:', self.video_thread.tag_pose_vecs.shape[0])
            self.video_thread.stop()  # 停止视频线程
        if self.robot_connection == 'control':
            self.rtde_c.stopScript()  # 停止机器人控制脚本
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """使用新的OpenCV图像更新image_label"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def update_tag_pose(self, tag_pose):
        if self.robot_connection is not False:
            self.tcp_force_vecs = np.concatenate((self.tcp_force_vecs, self.robot_thread.force.reshape([1,6])))
            print(self.robot_thread.force.reshape([1,6]))
        for i in range(6):
            self.tag_pose_labels[i].setText('%.1f'%(tag_pose[i]))

    def update_tcp_pose(self, tcp_pose):
        for i in range(3):
            self.tcp_pose_labels[i].setText('%.1f'%(tcp_pose[i]*1000))
        for i in range(3,6):
            self.tcp_pose_labels[i].setText('%.1f'%(tcp_pose[i]))
    
    def move_robot(self, tag_pose):
        s = time.time()
        speed_vector = self.get_speed()
        if speed_vector is not None and self.robot_connection == 'control':
            self.rtde_c.speedL(speed_vector, self.acceleration, 0.03)
            # print("Speed: ",speed_vector, time.time()-s)

    def get_speed(self):
        if self.video_thread.tag_pose_vecs.shape[0] < 100:
            return None
        # todo: 平滑处理
        win_size = 5


        temp_r0 = R.from_euler('zyx', self.video_thread.tag_pose_vecs[50:100, 5:2:-1], degrees=True)
        rpy0 = temp_r0.mean().as_euler('zyx', degrees=True)[::-1]
        p0 = np.concatenate((np.mean(self.video_thread.tag_pose_vecs[50:100, 0:3], axis=0), rpy0))
        
        temp_rt = R.from_euler('zyx', self.video_thread.tag_pose_vecs[-win_size:, 5:2:-1], degrees=True)
        rpy = temp_rt.mean().as_euler('zyx', degrees=True)[::-1]
        pt = np.concatenate((np.mean(self.video_thread.tag_pose_vecs[-win_size:, 0:3], axis=0), rpy))
        
        dp = pt - p0
        # print('original dp: ', dp)
        
        # 计算基于基座的TCP旋转
        robot_tcp_rt = np.array([float(self.tcp_pose_labels[0].text()), float(self.tcp_pose_labels[1].text()), float(self.tcp_pose_labels[2].text()),
               float(self.tcp_pose_labels[3].text()), float(self.tcp_pose_labels[4].text()), float(self.tcp_pose_labels[5].text())])
        bMe = (R.from_rotvec(robot_tcp_rt[3:6])).as_matrix()
        
        # 基于工具法兰的相机姿态
        eMc = R.from_euler('Z', 90, degrees=True).as_matrix()
        # 计算基于机器人基座的相机姿态
        bMc = np.dot(bMe, eMc)
        dp_translation = np.dot(bMc, dp[0:3])
        
        # 计算基于机器人基座的标记位姿差异
        init_pose = R.from_euler('zyx', rpy0[::-1], degrees=True)
        init_bMtag = np.dot(bMc, init_pose.as_matrix())
        init_euler = (R.from_matrix(init_bMtag)).as_euler('zyx', degrees=True)[::-1]
        
        current_pose = R.from_euler('zyx', rpy[::-1], degrees=True)
        current_bMtag = np.dot(bMc, current_pose.as_matrix())
        current_euler = (R.from_matrix(current_bMtag)).as_euler('zyx', degrees=True)[::-1]
        
        dp_rotation = current_euler - init_euler
        
        dp = np.concatenate((dp_translation, dp_rotation))
        # print('new dp: ', dp)

        # 控制器模式：旋转-位置
        if self.controller_selected == "rotation-position":
            roll, pitch, yaw = (dp[3:] >= 10).astype("float") + \
                    (np.logical_and(dp[3:] > 3, dp[3:] < 10)).astype("float")*(dp[3:]-3)/7 \
                    - (dp[3:] <= -10).astype("float") \
                    + (np.logical_and(dp[3:] < -3, dp[3:] > -10)).astype("float")*(dp[3:]+3)/7
            speed_vector = np.concatenate( ( self.speed_magnitude * np.array([pitch, -roll, yaw]), [0,0,0] ) )

        # 控制器模式：旋转-旋转
        elif self.controller_selected == "rotation-rotation":
            scale = (dp[3:] >= 10).astype("float") + \
                    (np.logical_and(dp[3:] > 3, dp[3:] < 10)).astype("float")*(dp[3:]-3)/7 \
                    - (dp[3:] <= -10).astype("float") \
                    + (np.logical_and(dp[3:] < -3, dp[3:] > -10)).astype("float")*(dp[3:]+3)/7
            speed_vector = np.concatenate( ( [0,0,0], self.speed_magnitude * np.array(scale) ) )
        
        # 控制器模式：位置-位置
        elif self.controller_selected == "position-position":
            dx, dy = (dp[:2] >= 5).astype("float") + \
                    (np.logical_and(dp[:2] > 0.5, dp[:2] < 5)).astype("float")*(dp[:2]-0.5)/4.5 \
                    - (dp[:2] <= -5).astype("float") \
                    + (np.logical_and(dp[:2] < -0.5, dp[:2] > -5)).astype("float")*(dp[:2]+0.5)/4.5
            drz = (dp[5] >= 10).astype("float") + \
                    (np.logical_and(dp[5] > 1, dp[5] < 10)).astype("float")*(dp[5]-1)/9 \
                    - (dp[5] <= -10).astype("float") \
                    + (np.logical_and(dp[5] < -1, dp[5] > -10)).astype("float")*(dp[5]+1)/9
            speed_vector = np.concatenate((self.speed_magnitude * np.array([dx, dy, drz]), [0, 0, 0]))

        # 控制器模式：位置-XY
        elif self.controller_selected == "position-xy":
            dx, dy = (dp[:2] >= 5).astype("float") + \
                    (np.logical_and(dp[:2] > 0.5, dp[:2] < 5)).astype("float")*(dp[:2]-0.5)/4.5 \
                    - (dp[:2] <= -5).astype("float") \
                    + (np.logical_and(dp[:2] < -0.5, dp[:2] > -5)).astype("float")*(dp[:2]+0.5)/4.5
            speed_vector = np.concatenate((self.speed_magnitude * np.array([dx, dy, 0]), [0, 0, 0]))

        # 控制器模式：位置-旋转
        elif self.controller_selected == "position-rotation":
            dx, dy = (dp[:2] >= 5).astype("float") + \
                    (np.logical_and(dp[:2] > 0.5, dp[:2] < 5)).astype("float")*(dp[:2]-0.5)/4.5 \
                    - (dp[:2] <= -5).astype("float") \
                    + (np.logical_and(dp[:2] < -0.5, dp[:2] > -5)).astype("float")*(dp[:2]+0.5)/4.5
            drz = (dp[5] >= 10).astype("float") + \
                    (np.logical_and(dp[5] > 1, dp[5] < 10)).astype("float")*(dp[5]-1)/9 \
                    - (dp[5] <= -10).astype("float") \
                    + (np.logical_and(dp[5] < -1, dp[5] > -10)).astype("float")*(dp[5]+1)/9
            speed_vector = np.concatenate( ( [0,0,0], self.rotation_magnitude * np.array([-dy, dx, -drz]) ) )
        
        else:
            speed_vector = np.concatenate( ( self.speed_magnitude*( (tvec > 5).astype("float") - (rpy < -5).astype("float") ), np.array([0,0,0]) ) )
        
        return speed_vector

    # 更改速度大小
    def change_speed(self, percent):
        self.speed_magnitude = MAX_SPEED * percent/100
        self.rotation_magnitude = MAX_ROTATION * percent/100

    # 更改控制器模式
    def change_controller(self, controller):
        self.controller_selected = controller

    # 将OpenCV图像转换为QPixmap格式
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

# 主程序入口
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
   