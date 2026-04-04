import cv2
import queue
import threading
import numpy as np
from scipy.spatial.transform import Rotation as spR
import glob
import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime

class SingleVideoCapture:
    """ Single camera """
    def __init__(self, cameraID=4, width=1920, height=1080):
        # Camera for marker
        self.cap = cv2.VideoCapture(cameraID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q = queue.Queue()
        self.flag = True

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.flag:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.flag = False
        self.cap.release()

"""Read Aruco code pose"""
class ReadAruCo:
    # Read Aruco pose from image
    def __init__(self) -> None:

        # 加载相机参数
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.014

    def readPose(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        if ids is None:
            return img, []

        color_image_result = img.copy()
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)
        pose_data = []
        for i in range(len(ids)):
            if ids[i] in [11, 12, 13, 14]:
                pose = np.hstack((ids[i], tvec[i].flatten(), rvec[i].flatten()))
                pose_data.append(pose)
            color_image_result = cv2.drawFrameAxes(color_image_result, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
            color_image_result = cv2.aruco.drawDetectedMarkers(color_image_result, corners, ids)
        
        return color_image_result, pose_data

if __name__ == "__main__":
    CAP = SingleVideoCapture(cameraID=8)
    MarkerDetector = ReadAruCo()

    count = 1
    max_count = 500  # 采集1000次
    resultpose = []

    while count <= max_count:
        img = CAP.read()
        result_img, pose_data = MarkerDetector.readPose(img)

        if pose_data:
            detected_ids = [p[0] for p in pose_data]
            required_ids = [11, 12, 13, 14]

            if all(req_id in detected_ids for req_id in required_ids):
                pose_data_sorted = sorted(pose_data, key=lambda x: required_ids.index(x[0]))
                id12_data = next((p for p in pose_data_sorted if p[0] == 12), None)

                if id12_data is not None:
                    translation = id12_data[1:4]
                    rotation = id12_data[4:7]
                    rotation_degrees = np.degrees(rotation)
                    GroundTruthPOSE = np.hstack((translation, rotation_degrees))

                    if len(resultpose) == 0:
                        resultpose = GroundTruthPOSE
                    else:
                        resultpose = np.vstack((resultpose, GroundTruthPOSE))

                    print("Transformed GroundTruthPOSE:", GroundTruthPOSE)
                else:
                    print("Error: ID 12 not found in detected IDs.")
        else:
            print("Error: Not enough valid pose data detected.")
        
        count += 1
        print("count:", count)

        cv2.imshow('Camera', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CAP.stop()
    cv2.destroyAllWindows()

    # 将 resultpose 重整为 nx6 的格式
    resultpose = np.reshape(resultpose, (-1, 6))

    # 保存到 CSV 文件
    csv_filename = 'resultpose.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(resultpose)

    print(f"Data saved to {csv_filename}")

    # 绘制并保存结果
    x, y, z, roll, pitch, yaw = [], [], [], [], [], []

    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))
            roll.append(float(row[3]))
            pitch.append(float(row[4]))
            yaw.append(float(row[5]))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(x, label='X Position')
    plt.plot(y, label='Y Position')
    plt.plot(z, label='Z Position')
    plt.title('Position Components Over Time')
    plt.xlabel('Count')
    plt.ylabel('Position (mm)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(roll, label='Roll (deg)')
    plt.plot(pitch, label='Pitch (deg)')
    plt.plot(yaw, label='Yaw (deg)')
    plt.title('Orientation Components Over Time')
    plt.xlabel('Count')
    plt.ylabel('Angle (degrees)')
    plt.legend()

    plt.tight_layout()

    img_folder = '/home/sen/Documents/InHand_pose/seeArucoAccurcyResultpose_plot'
    
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    output_file_path = os.path.join(img_folder, 'resultpose_plot_' + datetime.now().strftime("%m%d-%H%M%S") + '.png')
    plt.savefig(output_file_path)

    plt.show()
