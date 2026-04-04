import cv2

# 我们已经知道摄像头在索引 2 是可以打开的
cap = cv2.VideoCapture(8)

#cap = cv2.VideoCapture('/dev/video4')



if cap.isOpened():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法从摄像头捕获帧，退出循环")
                break

            # 显示帧
            cv2.imshow('Camera Frame', frame)
            #cv2.moveWindow('draw_result',1000,500)

            # 按 'q' 退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
else:
    print("无法打开摄像头")

