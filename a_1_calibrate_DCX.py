import numpy as np
import cv2

# Set the chessboard size and grid width
#chessboard_size = (10, 7)
#grid_width = 0.015

chessboard_size = (4, 3)
grid_width = 0.0187

# Prepare object points
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * grid_width

# Create arrays to store object points and image points
obj_points = [] # 3D points in real world space
img_points = [] # 2D points in image plane

# Initialize camera
cap = cv2.VideoCapture(10)  # 确保索引正确
if not cap.isOpened():
    print("无法打开视频源")
    exit(1)

# Set properties
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

cap.set(3,1920) #设置分辨率
cap.set(4,1080)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,3)

cv2.namedWindow('camera')

# Capture frames from camera
while True:
    ret, frame = cap.read()
    #frame = np.rot90(frame, 1)    # 对图像矩阵顺时针旋转90度

    if not ret:
        print("无法读取视频帧")
        continue

    # Find corners in current frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Draw corners on frame and display it
    if ret:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

    cv2.imshow('camera', frame)

    # Save image or exit loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            img_path = 'calibration_img\\calibration{}.jpg'.format(len(obj_points))
            cv2.imwrite(img_path, frame)
            print(f"Image saved: {img_path}")
    elif key == ord('q'):
        break

    # Calibrate camera if enough images are saved
    if len(obj_points) >= 50:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        mean_error = np.mean([cv2.norm(img_points[i], cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)[0], cv2.NORM_L2)/len(corners) for i in range(len(obj_points))])
        print(f"Reprojection error: {mean_error}")

        if mean_error < 1:
            np.savez("camera_params.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print("Camera calibrated")
            print("ret:",ret  )
            #print("mtx:\n",mtx)      # 内参数矩阵
            print("mtx:\n",np.array(mtx))
            print("dist畸变值:\n",dist   )  
            break
        else:
            print("Calibration failed, please try again")

# Release camera and close window
cap.release()
cv2.destroyAllWindows()