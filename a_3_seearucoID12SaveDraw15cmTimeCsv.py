import cv2
import numpy as np
import queue
import threading
import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv

class SingleVideoCapture:
    """ Single camera video capture management class. """
    def __init__(self, cameraID=0, width=1920, height=1080):
        self.cap = cv2.VideoCapture(cameraID)
        if not self.cap.isOpened():
            raise Exception(f"Camera {cameraID} could not be opened.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.q = queue.Queue(maxsize=10)  # Limit queue size to avoid excessive memory usage
        self.flag = True

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        """ Read frames as soon as they are available, keeping only most recent one. """
        while self.flag:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame.")
                break
            if self.q.full():
                try:
                    discarded_frame = self.q.get_nowait()  # discard previous (unprocessed) frame
                    #print("Discarded old frame.")
                except queue.Empty:
                    pass
            self.q.put(frame)
            #print("Frame put into queue.")

    def read(self):
        """ Retrieve the most recent frame. """
        try:
            frame = self.q.get(timeout=1)  # wait for up to 1 second for a frame
            print("Frame retrieved from queue.")
            return frame
        except queue.Empty:
            print("Queue is empty, no frame to retrieve.")
            return None

    def stop(self):
        """ Stop the video capture. """
        self.flag = False
        self.cap.release()


class ReadAruCo:
    """ Aruco code pose reader using a single camera. """
    def __init__(self):
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.mark_size = 0.016
        self.frame_count = 0

    def readPose(self, img, save_folder):
        """ Read and display Aruco marker pose from a single image. """
        if img is None:
            print("Image is None.")
            return img
        #img = np.rot90(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        gray = cv2.filter2D(gray, -1, kernel)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        if ids is None:
            print("No Aruco IDs detected.")
            return img

        img_with_aruco = img.copy()
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)
        tvec*=(-100)  # m -> cm
         
        for i in range(len(ids)):
            if ids[i] == 12:  # Process only marker with ID 12
                img_with_aruco = cv2.aruco.drawAxis(img_with_aruco, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
                img_with_aruco = cv2.aruco.drawDetectedMarkers(img_with_aruco, corners, ids)
                print(f"ID 12: Rotation Vector (rvec) = {rvec[i]}, Translation Vector (tvec) = {tvec[i]}")

                # Save the adjusted second value of the translation vector and the timestamp to a file
                adjusted_value = tvec[i][0][1] + 40.0  # Subtract 0.10 from the second value (y component) of the Translation Vector
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                file_name_txt = "{:08d}.txt".format(self.frame_count)
                file_path = os.path.join(save_folder, file_name_txt)
                with open(file_path, 'w') as f:
                    f.write(f"{timestamp},{adjusted_value}\n")
                self.frame_count += 1

        return img_with_aruco


def save_height_data_to_csv(data_folder, csv_filename):
    """ Read stored data and save to a CSV file. """
    timestamps = []
    heights = []
    filenames = sorted([f for f in os.listdir(data_folder) if f.endswith('.txt')])

    for filename in filenames:
        file_path = os.path.join(data_folder, filename)
        try:
            with open(file_path, 'r') as f:
                line = f.readline().strip()
                timestamp, height = line.split(',')
                timestamps.append(timestamp)
                heights.append(float(height))
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")

    if not heights:
        print("No height data found to save.")
        return

    # Convert timestamps to relative times in seconds
    base_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
    relative_times = [(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') - base_time).total_seconds() for ts in timestamps]

    # Save data to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time (s)', 'Height (m)'])
        for rel_time, height in zip(relative_times, heights):
            writer.writerow([rel_time, height])
    print(f"Height data saved to {csv_filename}.")


def plot_height_changes(data_folder):
    """ Read stored data and plot height changes over time. """
    timestamps = []
    heights = []
    filenames = sorted([f for f in os.listdir(data_folder) if f.endswith('.txt')])

    for filename in filenames:
        file_path = os.path.join(data_folder, filename)
        try:
            with open(file_path, 'r') as f:
                line = f.readline().strip()
                timestamp, height = line.split(',')
                timestamps.append(timestamp)
                heights.append(float(height))
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")

    if not heights:
        print("No height data found to plot.")
        return

    # Convert timestamps to relative times in seconds
    base_time = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S')
    relative_times = [(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') - base_time).total_seconds() for ts in timestamps]

    # Plot
    plt.figure()
    plt.plot(relative_times, heights, marker='o', label='Height (cm)')
    plt.title('Height Changes Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (cm)')
    plt.ylim(-1,12)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(data_folder, 'Height_Changes_Over_Time.png'))
    plt.show()


if __name__ == "__main__":
    video_capture = SingleVideoCapture(cameraID=8)  # Adjust camera ID based on your setup
    
    img_folder = "./ArcucoPOSE"
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    hightpose_folder = f"HightPose_{timestamp}"

    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    hightpose_path = os.path.join(img_folder, hightpose_folder)
    if not os.path.isdir(hightpose_path):
        os.mkdir(hightpose_path)


    aruco_reader = ReadAruCo()

    while True:
        img = video_capture.read()
        if img is None:
            print("No frame retrieved.")
            continue
        result_img = aruco_reader.readPose(img, hightpose_path)  # Pass the save folder path
        cv2.imshow('Camera', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.stop()
    cv2.destroyAllWindows()

    # Save height data to CSV
    csv_filename = os.path.join(hightpose_path, 'height_data.csv')
    save_height_data_to_csv(hightpose_path, csv_filename)

    # Plot height changes after the loop ends
    plot_height_changes(hightpose_path)
