import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 文件夹路径
folder_path = '/home/sen/Documents/InHand_pose/IMG_DATA_LS/IMG_DATA_PEACH_241227/POSE_TXT_filtering'

# 获取文件列表（按照文件名排序）
file_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')])

# 初始化数据结构
data_by_id = {}  # 按 ID 分类的数据（用于绘图）
all_data = []  # 只保存 ID=12 的数据

# 读取所有文件数据
for file_path in file_list:
    data = np.loadtxt(file_path)
    for row in data:
        id_ = int(row[0])  # 获取 ID
        if id_ == 12:  # 只保留 ID=12 的数据
            all_data.append(row[1:])  # 只保留 x, y, z, rx, ry, rz 数据

        # 按 ID 分类数据，用于绘图
        if id_ not in data_by_id:
            data_by_id[id_] = {'x': [], 'y': [], 'z': [], 'rx': [], 'ry': [], 'rz': []}
        data_by_id[id_]['x'].append(row[1])
        data_by_id[id_]['y'].append(row[2])
        data_by_id[id_]['z'].append(row[3])
        data_by_id[id_]['rx'].append(row[4])
        data_by_id[id_]['ry'].append(row[5])
        data_by_id[id_]['rz'].append(row[6])

# 绘制每个 ID 的数据
for id_, data in data_by_id.items():
    plt.figure(figsize=(12, 8))

    # 绘制位置分量
    plt.subplot(2, 1, 1)
    plt.plot(data['x'], label='X')
    plt.plot(data['y'], label='Y')
    plt.plot(data['z'], label='Z')
    plt.title(f'Position Components for ID {id_}')
    plt.xlabel('Sequence Number')
    plt.ylabel('Position (mm)')
    plt.legend()
    plt.grid(True)

    # 绘制姿态分量
    plt.subplot(2, 1, 2)
    plt.plot(data['rx'], label='Roll')
    plt.plot(data['ry'], label='Pitch')
    plt.plot(data['rz'], label='Yaw')
    plt.title(f'Orientation Components for ID {id_}')
    plt.xlabel('Sequence Number')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 将 ID=12 的数据保存为 CSV 文件
df = pd.DataFrame(all_data, columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
output_file = 'IMG_DATA_PEACH_241227_objectPose_data_ID12.csv'
df.to_csv(output_file, index=False)

print(f"Data for ID=12 saved to {output_file}")
