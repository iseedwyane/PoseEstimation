import cv2

# 循环尝试打开可能的摄像头索引
for index in range(20):  # 假设系统中不会超过10个摄像头
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"成功打开索引为 {index} 的摄像头")
        cap.release()
        #break
else:
    print("未能打开任何摄像头")

#
#lsusb
#检查所有连接的视频设备及其相关的设备文件
#v4l2-ctl --list-devices
#来检查设备文件的权限
#ls -l /dev/video4
#sudo usermod -a -G video $USER
#cv2.namedWindow("Frame")