import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as spR
import matplotlib.pyplot as plt
from datetime import datetime

class ArucoDataCollector:
    def __init__(self, camera_id=10, width=1920, height=1080, mark_size=0.016):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.camera_matrix = np.array([[1.35827380e+03, 0.00000000e+00, 9.64499616e+02],
                                       [0.00000000e+00, 1.35569983e+03, 5.67785628e+02],
                                       [0., 0., 1.]])
        self.camera_dist = np.array([0.0433853, -0.05228565, 0.00079905, 0.00208749, -0.01949841])
        self.mark_size = mark_size
        self.error_data = []

    def collect_data(self, num_samples=1000):
        sample_count = 0
        while sample_count < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture image")
                continue
            
            pose_data, result_img = self.detect_aruco_markers(frame)
            if pose_data is not None and pose_data.size > 0:
                self.error_data.append(pose_data)
                sample_count += 1
                cv2.imshow('Aruco Detection', result_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No Aruco markers detected")

        self.cap.release()
        cv2.destroyAllWindows()

    def detect_aruco_markers(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        if ids is None or len(ids) != 4:
            return None, img
        
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.mark_size, self.camera_matrix, self.camera_dist)
        pose_data = np.hstack((ids, tvec.reshape(-1, 3), rvec.reshape(-1, 3)))

        for i in range(len(ids)):
            cv2.aruco.drawAxis(img, self.camera_matrix, self.camera_dist, rvec[i], tvec[i], self.mark_size)
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
        
        return pose_data, img

    def compute_errors(self):
        ids = [11, 12, 13, 14]
        errors = {id_: {'translation': [], 'rotation': []} for id_ in ids}

        for sample in self.error_data:
            for pose in sample:
                id_, tx, ty, tz, rx, ry, rz = pose
                if id_ in errors:
                    # 转换单位：平移单位为毫米，旋转单位为度
                    errors[id_]['translation'].append([tx * 1000, ty * 1000, tz * 1000])  # 转换为毫米
                    errors[id_]['rotation'].append(np.degrees([rx, ry, rz]))  # 转换为度

        error_results = {}
        for id_ in ids:
            translations = np.array(errors[id_]['translation'])
            rotations = np.array(errors[id_]['rotation'])
            translation_mean = np.mean(translations, axis=0)
            translation_std = np.std(translations, axis=0)
            rotation_mean = np.mean(rotations, axis=0)
            rotation_std = np.std(rotations, axis=0)
            error_results[id_] = {
                'translation_mean': translation_mean,
                'translation_std': translation_std,
                'rotation_mean': rotation_mean,
                'rotation_std': rotation_std
            }

        return error_results

    def save_errors_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for id_, error in self.compute_errors().items():
            file_path = os.path.join(output_folder, f"error_id_{int(id_)}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Translation Mean (mm): {error['translation_mean']}\n")
                f.write(f"Translation Std Dev (mm): {error['translation_std']}\n")
                f.write(f"Rotation Mean (degrees): {error['rotation_mean']}\n")
                f.write(f"Rotation Std Dev (degrees): {error['rotation_std']}\n")

    def plot_histograms(self, output_folder):
        data = self.compute_errors()
        ids = [11, 12, 13, 14]
        components = ['X', 'Y', 'Z']

        # Create figure and axis for plotting
        fig, axs = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle('Aruco Marker Pose Estimation Errors', fontsize=16)

        bar_width = 0.2  # Width of each bar
        positions = np.arange(len(components))  # Positions for each set of bars

        for i, id_ in enumerate(ids):
            error = data[id_]
            
            # Plot Translation Errors (in mm)
            for j in range(3):
                axs[0, j].bar(positions + i * bar_width, error['translation_mean'][j], width=bar_width, label=f'ID {id_}')
                axs[0, j].set_title(f'Translation {components[j]} Error (mm)')
                axs[0, j].set_xticks(positions + bar_width * (len(ids) - 1) / 2)
                axs[0, j].set_xticklabels(components)
                axs[0, j].legend()

            # Plot Rotation Errors (in degrees)
            for j in range(3):
                axs[1, j].bar(positions + i * bar_width, error['rotation_mean'][j], width=bar_width, label=f'ID {id_}')
                axs[1, j].set_title(f'Rotation {components[j]} Error (degrees)')
                axs[1, j].set_xticks(positions + bar_width * (len(ids) - 1) / 2)
                axs[1, j].set_xticklabels(components)
                axs[1, j].legend()

        # Adjust layout and save the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(output_folder, 'histogram_all_ids.png')
        plt.savefig(plot_path)
        plt.close(fig)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"./aruco_error_data/try_{timestamp}"
    
    collector = ArucoDataCollector()
    collector.collect_data(num_samples=100)
    collector.save_errors_to_file(output_folder)
    collector.plot_histograms(output_folder)
    
    print("Data collection and error calculation complete.")
