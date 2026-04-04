import os
import csv

# 定义要读取的文件夹路径
folder_path = "/home/sen/Desktop/POSE_TXT"  # 你提供的文件夹路径

# 存储ID=12的pose数据
pose_data = []

# 遍历文件夹中的所有txt文件
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            # 读取文件中的每一行
            lines = file.readlines()
            for line in lines:
                # 每行数据格式是 ID, x, y, z, Rx, Ry, Rz
                data = line.strip().split()
                if len(data) == 7 and data[0] == "12":  # 只提取ID为12的行
                    # 提取出ID=12的x, y, z, Rx, Ry, Rz并存储
                    pose_data.append([float(i) for i in data[1:]])

# 检查是否提取到数据
if pose_data:
    # 保存为CSV文件
    output_file = "/home/sen/Desktop/pose_vecs.csv"  # 输出的CSV文件路径
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头
        csv_writer.writerow(['x', 'y', 'z', 'Rx', 'Ry', 'Rz'])
        # 写入数据
        csv_writer.writerows(pose_data)
    print(f"Pose data for ID=12 has been saved to {output_file}")
else:
    print("No data found for ID=12.")
