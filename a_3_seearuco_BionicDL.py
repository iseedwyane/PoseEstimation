import cv2
import numpy as np
import csv

capID = 10

def process_rvec(rvec1):
    """
    处理 rvec1 向量，如果第一个元素为负数，则将前两个元素取反。
    """
    if rvec1[0] < 0:
        rvec1[:2] *= -1
    return rvec1

def process_euler1(rvec1):
    """
    处理 rvec1 向量，如果第一个元素为负数，则将前两个元素取反。
    """
    if rvec1[0] < 0:
        rvec1[0] += 360
    return rvec1

def optimized_binary_threshold(img, black_pixel_ratio=0.5, preset_threshold=None, sample_ratio=0.25):
    """
    对图像进行优化的二值化处理，可以从预设阈值开始搜索。

    Args:
        img: 输入图像（假设为灰度图）。
        black_pixel_ratio: 目标黑色像素比例，默认为0.5。
        preset_threshold: 预设的阈值，默认为None。如果提供，将从此值开始搜索。
        sample_ratio: 下采样比例，默认为0.25。

    Returns:
        二值化后的图像。
    """
    # 下采样
    if sample_ratio < 1:
        height, width = img.shape[:2]
        small_img = cv2.resize(img, (int(width * sample_ratio), int(height * sample_ratio)))
    else:
        small_img = img

    # 计算目标黑色像素数量
    total_pixels = small_img.size
    target_black_pixels = int(total_pixels * black_pixel_ratio)

    # 从预设阈值开始搜索
    if preset_threshold is not None:
        threshold = preset_threshold
        black_pixels = np.sum(small_img <= threshold)

        if black_pixels < target_black_pixels:
            # 如果预设阈值太低，向上搜索
            for t in range(threshold + 1, 256):
                black_pixels = np.sum(small_img <= t)
                if black_pixels >= target_black_pixels:
                    threshold = t
                    break
        else:
            # 如果预设阈值太高，向下搜索
            for t in range(threshold - 1, -1, -1):
                black_pixels = np.sum(small_img <= t)
                if black_pixels < target_black_pixels:
                    threshold = t + 1
                    break
    else:
        # 如果没有预设阈值，使用完整搜索
        threshold = 0
        for t in range(256):
            black_pixels = np.sum(small_img <= t)
            if black_pixels >= target_black_pixels:
                threshold = t
                break
    print("threshold=",threshold)

    # 使用阈值进行二值化
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return thresh

def binary_threshold_for_black_pixel_ratio(img, black_pixel_ratio=0.5):
    """
    对图像进行二值化，使得黑色像素占总像素的比例接近指定值。

    Args:
        image_path: 图像路径。
        black_pixel_ratio: 黑色像素占总像素的比例，默认为 0.5。

    Returns:
        二值化后的图像。
    """
    # 计算图像直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 计算目标黑色像素数量
    total_pixels = img.shape[0] * img.shape[1]
    target_black_pixels = int(total_pixels * black_pixel_ratio)

    # 找到合适的阈值
    threshold = 0
    sum_pixels = 0
    for i in range(256):
        sum_pixels += hist[i][0]
        if sum_pixels >= target_black_pixels:
            threshold = i
            break

    # 使用阈值进行二值化
    ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return thresh

f = open("test.csv", mode='w')
writer = csv.writer(f)

# Define ArUco dictionary and parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

# 初始化数据收集列表
collected_data = []
frame_count = 0
total_frames = 2000

cap = cv2.VideoCapture(capID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 加载相机参数
matrix = np.load('camera_params_outside1920x1080LiS.npz')
mtx = matrix['mtx']
dist = matrix['dist']

while frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #thresh = optimized_binary_threshold(gray, black_pixel_ratio=0.28, preset_threshold=85)
    thresh = gray
    corners, ids, rejected = cv2.aruco.detectMarkers(thresh, aruco_dict, parameters=aruco_params)

    if ids is not None:
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.016, mtx, dist)
            rvec1 = np.squeeze(rvec)
            tvec1 = np.squeeze(tvec)
            R, _ = cv2.Rodrigues(rvec)
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((R, np.zeros((3, 1)))))
            euler_angles = process_euler1(euler_angles)
            euler_angles = np.deg2rad(euler_angles)
            data = np.hstack((euler_angles.flatten(), tvec1))
            collected_data.append(data)
            frame_count += 1

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('thresh', thresh)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

    print(f"Collected frames: {frame_count}/{total_frames}")

cap.release()
cv2.destroyAllWindows()

# 处理收集到的数据
if collected_data:
    collected_data = np.array(collected_data)
    dimensions = collected_data.shape[1]

    results = []
    for i in range(dimensions):
        dim_data = collected_data[:, i]
        range_val = np.max(dim_data) - np.min(dim_data)
        mean_val = np.mean(dim_data)
        var_val = np.var(dim_data)
        results.append({
            'dimension': i + 1,
            'range': range_val,
            'mean': mean_val,
            'variance': var_val
        })
    
    # 打印结果
    print("\nResults:")
    for result in results:
        print(f"Dimension {result['dimension']}:")
        print(f"  Range: {result['range']}")
        print(f"  Mean: {result['mean']}")
        print(f"  Variance: {result['variance']}")
        print()

    # 保存结果到CSV文件
    with open('aruco_stats.csv', 'w', newline='') as csvfile:
        fieldnames = ['dimension', 'range', 'mean', 'variance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print("Results saved to aruco_stats.csv")
else:
    print("No data collected.")
