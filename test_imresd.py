import cv2

# 读取图像
img_outside = cv2.imread('./IMG_DATA_LS/IMG_DATA_BASEBALL_0628/IMG_OUTSIDE/00000000.jpg')

if img_outside is None:
    print("Error: Could not read the outside image.")
else: 
    print("read the outside image.")  
    print(img_outside.shape)

    # 在主线程中显示图像
    def show_image():
        cv2.imshow('img_outside', img_outside)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 使用一个简单的计时器来延迟图像显示
    from threading import Timer
    Timer(0.1, show_image).start()
    print("NoError: Could imshow the outside image.")