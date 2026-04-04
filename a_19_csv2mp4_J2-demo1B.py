import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os
import numpy as np

# 读取CSV文件
current_dir = os.getcwd()
print(current_dir)
filename = os.path.join(current_dir, 'closeloopResuts', 'demo1_processed_data_final.csv')

# 读取文件并跳过第一行
data = pd.read_csv(filename, skiprows=1, header=None)

# 打印数据信息以便调试
print("数据形状:", data.shape)
print("列名:", data.columns.tolist())
print("前几行数据:")
print(data.head())

# 提取时间和数据列
time_seconds = data.iloc[:, 0]  # 第0列：时间数据（秒）
y1 = data.iloc[:, 4]+2  # 第4列：GroundTruth_mm
y2 = data.iloc[:, 5]+2  # 第5列：Predict_mm
refH = 75 * pd.Series([1] * len(data))  # 创建参考高度列

print("时间范围:", time_seconds.min(), "到", time_seconds.max(), "秒")
print("GroundTruth范围:", y1.min(), "到", y1.max(), "mm")
print("Predict范围:", y2.min(), "到", y2.max(), "mm")

# 设置合适的坐标轴范围
x_min = time_seconds.min()
x_max = time_seconds.max()
# y_min = min(y1.min(), y2.min(), 75) - 10
# y_max = max(y1.max(), y2.max(), 75) + 10
y_min = 20
y_max = 120


# 设置视频保存路径
output_video = 'demo1_processed_data_final_B.mp4'

# 创建图形
fig, ax = plt.subplots(figsize=(12, 7))

# 初始化图形
line1, = ax.plot([], [], color='#778899', label='Ref Height', linestyle='--', linewidth=2)
line2, = ax.plot([], [], color='#DC143C', label='Ground Truth', linewidth=2)
line3, = ax.plot([], [], color='#483D8B', label='Predict Value', linewidth=2)

# 设置坐标轴范围（重要！）
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Height (mm)', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True)
#ax.set_title('Height Comparison Animation')

# 更新函数：每一帧更新图形
def update(frame):
    if frame > 0:
        current_time = time_seconds[:frame]
        line1.set_data(current_time, refH[:frame])  # 更新参考值
        line2.set_data(current_time, y1[:frame])    # 更新实际值
        line3.set_data(current_time, y2[:frame])    # 更新预测值
    return line1, line2, line3

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(time_seconds), 
                             interval=50, blit=True, repeat=False)

# 保存动画为 MP4 文件
print("正在保存动画...")
writer = FFMpegWriter(fps=15, bitrate=5000)
ani.save(output_video, writer=writer, dpi=150)
print(f"动画已保存为: {output_video}")

# # 显示图形
# plt.tight_layout()
# plt.show()

# # 同时绘制静态图进行检查
# plt.figure(figsize=(12, 6))
# plt.plot(time_seconds, y1, color='#DC143C', label='Ground Truth', linewidth=2)
# plt.plot(time_seconds, y2, color='#483D8B', label='Predict Value', linewidth=2)
# plt.plot(time_seconds, refH, color='#778899', label='Ref (75 mm)', linestyle='--', linewidth=2)
# plt.xlabel('Time (s)', fontsize=12)
# plt.ylabel('Height (mm)', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.title('Height Comparison - Static Plot')
# plt.tight_layout()
# plt.show()