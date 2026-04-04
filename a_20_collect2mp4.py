import cv2
import os

# 图片文件夹路径
image_folder = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_MASTERBALL_0907/IMG_OUTSIDE/'  # 替换为图片所在的路径
output_video = 'output_video_IMG_DATA_MASTERBALL_0907_collectting.mp4'     # 输出视频文件名

# 获取所有图片文件并排序
images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 确保按文件名排序
images.sort()

# 确保文件夹中有图片
if not images:
    print("该文件夹中没有图片！")
    exit()

# 读取第一张图片以获取宽度和高度
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# 定义视频编码格式和视频输出
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以选择不同的编码格式
video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  # 30是帧率

# 将每一张图片写入视频
for image in images:
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)

    # 确保图片尺寸一致
    img = cv2.resize(img, (width, height))  # 如果尺寸不一致，可以调整大小

    video.write(img)

# 释放视频文件
video.release()

print(f"视频已保存为 {output_video}")
