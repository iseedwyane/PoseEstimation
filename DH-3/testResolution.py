import cv2

cap = cv2.VideoCapture(4)

# 尝试一些常见分辨率
resolutions = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]

for width, height in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Requested: {width}x{height}, Actual: {actual_width}x{actual_height}")

cap.release()