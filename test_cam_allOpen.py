import cv2

# 循环尝试打开可能的摄像头索引
for index in range(20):  # 假设系统中不会超过20个摄像头
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"成功打开索引为 {index} 的摄像头")
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break

            cv2.imshow("Frame", frame)

            # 按 'q' 键退出显示
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"无法打开索引为 {index} 的摄像头")
else:
    print("未能打开任何摄像头")

# 释放资源
cv2.destroyAllWindows()


#
#lsusb
#检查所有连接的视频设备及其相关的设备文件
#v4l2-ctl --list-devices
#来检查设备文件的权限
#ls -l /dev/video4
#sudo usermod -a -G video $USER
#cv2.namedWindow("Frame")